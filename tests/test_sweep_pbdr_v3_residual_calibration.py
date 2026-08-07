from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiments import pbdr_v3_residual_calibration as calibration
from experiments import pbdr_v4_internal_cache as cache_io
from experiments import pbdr_v4_metric_core as metric_core
from experiments import pbdr_v4_split_authority as split_authority
from experiments import sweep_pbdr_v3_residual_calibration as sweep


DATASET = "NUAA-SIRST"
ROLE = "best_miou"
SAMPLE_IDS = ("synthetic-internal-validation-0",)
SOURCE_LOCK_SHA256 = "e" * 64


def _synthetic_projection() -> dict[str, object]:
    projection: dict[str, object] = {
        "schema": split_authority.SCHEMA,
        "status": "synthetic_split_projection",
        "dataset_order": [DATASET],
        "model_selection_only": True,
        "parent_seen_official_train": True,
        "official_test_accessed": False,
        "split_reconstruction_performed": False,
        "datasets": {
            DATASET: {
                "dataset": DATASET,
                "canonical_split_sha256": "1" * 64,
                "counts": {
                    "official_train": 1,
                    "development_train": 1,
                    "internal_validation": len(SAMPLE_IDS),
                },
                "ordered_id_sha256": {
                    "official_train_ids": cache_io.canonical_sha256(
                        list(SAMPLE_IDS)
                    ),
                    "development_train_ids": cache_io.canonical_sha256(
                        list(SAMPLE_IDS)
                    ),
                    "internal_validation_ids": cache_io.canonical_sha256(
                        list(SAMPLE_IDS)
                    ),
                },
                "model_selection_only": True,
                "parent_seen_official_train": True,
                "official_test_accessed": False,
            }
        },
    }
    projection["projection_sha256"] = cache_io.canonical_sha256(projection)
    return projection


def _checkpoint(name: str, token: str) -> cache_io.CheckpointBinding:
    return cache_io.CheckpointBinding(
        path=f"/synthetic/{name}.pth.tar",
        bytes=100 + len(name),
        file_sha256=token * 64,
        state_sha256=token * 64,
    )


def _sample_arrays() -> dict[str, np.ndarray]:
    # Every grid entry has the same binary prediction.  A -0.15 bias improves
    # BCE over the Current anchor by only ~4e-8, so the winner demonstrates
    # strict role-key improvement with no positive-effect-size threshold.
    base = np.full((3, 3), np.float32(-15.0), dtype=np.float32)
    base[1, 1] = np.float32(15.0)
    base = np.ascontiguousarray(base)
    delta = np.zeros((3, 3), dtype=np.float32, order="C")
    routed = np.add(base, delta, dtype=np.float32)
    target = np.zeros((3, 3), dtype=np.float32, order="C")
    target[1, 1] = np.float32(1.0)
    return {
        "base_logits": base,
        "delta_logits": delta,
        "routed_logits": np.ascontiguousarray(routed),
        "current_logits": base.copy(order="C"),
        "original_logits": base.copy(order="C"),
        "target": np.ascontiguousarray(target),
    }


class PBDRV3ResidualCalibrationSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls._cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.projection = _synthetic_projection()
        cls.cache_path = cls.root / "cache"
        metric_sha = cache_io.file_sha256(Path(metric_core.__file__).resolve())
        with cache_io.InternalRawLogitCacheWriter(
            cls.cache_path,
            dataset_name=DATASET,
            parent_role=ROLE,
            partition="internal_validation",
            split_projection=cls.projection,
            ordered_sample_ids=SAMPLE_IDS,
            v3_checkpoint=_checkpoint("v3", "a"),
            current_checkpoint=_checkpoint("current", "b"),
            original_checkpoint=_checkpoint("original", "c"),
            normalization={"mean": 10.0, "std": 2.0},
            metric_core_sha256=metric_sha,
            source_lock_sha256=SOURCE_LOCK_SHA256,
        ) as writer:
            writer.append_sample(
                sample_id=SAMPLE_IDS[0],
                height=3,
                width=3,
                **_sample_arrays(),
            )
            writer.finalize()
        cls.cache = cache_io.read_cache(
            cls.cache_path,
            split_projection=cls.projection,
        )
        cls.result = sweep.compute_sweep_result(cls.cache)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()
        torch.backends.cuda.matmul.allow_tf32 = cls._matmul_tf32
        torch.backends.cudnn.allow_tf32 = cls._cudnn_tf32

    def test_all_378_configs_anchors_and_strict_zero_margin_selection(self) -> None:
        result = self.result
        self.assertEqual(result["candidate_count"], 378)
        self.assertEqual(len(result["candidates"]), 378)
        self.assertEqual(
            [entry["grid_index"] for entry in result["candidates"]],
            list(range(378)),
        )
        self.assertEqual(
            len(
                {
                    json.dumps(entry["config"], sort_keys=True)
                    for entry in result["candidates"]
                }
            ),
            378,
        )
        for entry in result["candidates"]:
            self.assertEqual(
                set(entry),
                {
                    "name",
                    "family",
                    "metrics",
                    "exact_sufficient_statistics",
                    "role_key",
                    "grid_index",
                    "config",
                },
            )
            self.assertEqual(len(entry["role_key"]["components"]), 6)
            self.assertEqual(len(entry["exact_sufficient_statistics"]), 8)

        current_index = result["grid_binding"]["current_anchor_grid_index"]
        v3_index = result["grid_binding"]["v3_anchor_grid_index"]
        self.assertEqual(current_index, 3)
        self.assertEqual(v3_index, 147)
        self.assertEqual(
            result["candidates"][current_index]["metrics"],
            result["baselines"]["current"]["metrics"],
        )
        self.assertEqual(
            result["candidates"][v3_index]["metrics"],
            result["baselines"]["v3_anchor"]["metrics"],
        )
        self.assertTrue(
            result["anchor_replay"]["current"]["byte_exact_for_every_sample"]
        )
        self.assertTrue(
            result["anchor_replay"]["v3"]["byte_exact_for_every_sample"]
        )

        selected_index = result["selected"]["grid_index"]
        self.assertEqual(selected_index, 0)
        selected = result["candidates"][selected_index]
        current = result["candidates"][current_index]
        self.assertEqual(
            selected["role_key"]["components"][:-1],
            current["role_key"]["components"][:-1],
        )
        gain = (
            current["metrics"]["test_loss"]
            - selected["metrics"]["test_loss"]
        )
        self.assertGreater(gain, 0.0)
        self.assertLess(gain, 1.0e-6)
        self.assertEqual(
            result["selection_policy"]["comparison"],
            "strict_lexicographic_full_role_key_no_positive_margin",
        )
        self.assertEqual(
            result["selection_policy"]["exact_tie_break"],
            "earlier_grid_index",
        )

        self.assertEqual(
            result["source_binding"]["source_lock_sha256"],
            SOURCE_LOCK_SHA256,
        )
        self.assertFalse(result["official_test_accessed"])
        self.assertFalse(
            result["runtime_binding"]["live_cuda_matmul_allow_tf32"]
        )
        self.assertFalse(result["runtime_binding"]["live_cudnn_allow_tf32"])
        self.assertEqual(
            result["probability_contract"]["threshold"],
            {"numerator": 1, "denominator": 2},
        )

    def test_oexcl_commit_read_replay_tamper_and_symlink_rejection(self) -> None:
        destination = self.root / "sweep-result.json"
        committed = sweep.write_sweep_result_once(
            destination,
            result=self.result,
            cache=self.cache,
        )
        self.assertEqual(committed, destination.resolve())
        replayed = sweep.read_sweep_result(
            destination,
            cache_path=self.cache_path,
            split_projection=self.projection,
        )
        self.assertEqual(replayed, self.result)
        with self.assertRaises(FileExistsError):
            sweep.write_sweep_result_once(
                destination,
                result=self.result,
                cache=self.cache,
            )

        symlink = self.root / "result-symlink.json"
        os.symlink(destination, symlink)
        with self.assertRaises(FileExistsError):
            sweep.write_sweep_result_once(
                symlink,
                result=self.result,
                cache=self.cache,
            )
        with self.assertRaisesRegex(
            sweep.PBDRV3ResidualSweepError,
            "regular non-symlink",
        ):
            sweep.read_sweep_result(
                symlink,
                cache_path=self.cache_path,
                split_projection=self.projection,
            )

        tampered = copy.deepcopy(self.result)
        tampered["selected"]["grid_index"] = 1
        del tampered["result_sha256"]
        tampered["result_sha256"] = cache_io.canonical_sha256(tampered)
        tampered_path = self.root / "tampered-result.json"
        tampered_path.write_bytes(
            cache_io.canonical_json_bytes(tampered, trailing_newline=True)
        )
        with self.assertRaisesRegex(
            sweep.PBDRV3ResidualSweepError,
            "full cache-backed replay",
        ):
            sweep.read_sweep_result(
                tampered_path,
                cache_path=self.cache_path,
                split_projection=self.projection,
            )

    def test_internal_validation_only_and_both_anchor_mismatches_fail(self) -> None:
        forged_manifest = copy.deepcopy(dict(self.cache.manifest))
        forged_manifest["identity"]["partition"] = "development_train"
        forged = cache_io.ValidatedInternalRawLogitCache(
            path=self.cache.path,
            manifest=forged_manifest,
            samples=self.cache.samples,
        )
        with self.assertRaisesRegex(
            sweep.PBDRV3ResidualSweepError,
            "internal_validation only",
        ):
            sweep.compute_sweep_result(forged)

        original_apply = calibration.apply_residual_calibration

        for anchor, message in (
            (calibration.CURRENT_ANCHOR, "Current anchor"),
            (calibration.PBDR_V3_ANCHOR, "PBDR-V3 anchor"),
        ):
            def corrupt(
                base: torch.Tensor,
                delta: torch.Tensor,
                config: calibration.ResidualCalibration,
                *,
                selected_anchor: calibration.ResidualCalibration = anchor,
            ) -> torch.Tensor:
                value = original_apply(base, delta, config)
                if config == selected_anchor:
                    value = value.clone()
                    value[0, 0, 0, 0] += torch.tensor(
                        1.0e-3,
                        dtype=value.dtype,
                    )
                return value

            with self.subTest(anchor=anchor.name), mock.patch.object(
                calibration,
                "apply_residual_calibration",
                side_effect=corrupt,
            ):
                with self.assertRaisesRegex(
                    sweep.PBDRV3ResidualSweepError,
                    message,
                ):
                    sweep.compute_sweep_result(self.cache)

    def test_module_has_no_dataset_access_or_positive_margin_gate(self) -> None:
        source = Path(sweep.__file__).resolve().read_text(encoding="utf-8")
        for forbidden in (
            "load_index",
            "stratified_split",
            "img_idx",
            "official_test_loader",
            "passed_gate",
            "min_delta",
            "epsilon",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
