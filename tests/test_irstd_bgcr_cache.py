"""Synthetic/static contracts for the train-only IRSTD BGCR cache.

No test in this file imports a dataset/evaluator, opens a repository index, or
constructs a production model.  Production checkpoint/data verification is
covered by fail-closed constants and helpers in the cache module itself.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from experiments import cache_irstd_frozen_context_v1 as cache


def _synthetic_arrays() -> dict[str, np.ndarray]:
    spatial = (cache.FULL_CONTEXT_HEIGHT, cache.FULL_CONTEXT_WIDTH)
    arrays: dict[str, np.ndarray] = {
        "image": np.zeros((1, *spatial), dtype=np.float32),
        "target": np.zeros((1, *spatial), dtype=np.float32),
        "u1": np.zeros((32, *spatial), dtype=np.float32),
        "z_out": np.zeros((1, *spatial), dtype=np.float32),
        "z_d0": np.zeros((1, *spatial), dtype=np.float32),
        "z_gt2": np.zeros((1, *spatial), dtype=np.float32),
        "z_gt3": np.zeros((1, *spatial), dtype=np.float32),
        "z_gt4": np.zeros((1, *spatial), dtype=np.float32),
        "z_gt5": np.zeros((1, *spatial), dtype=np.float32),
        "baseline1000_logits": np.zeros((1, *spatial), dtype=np.float32),
        "target_component_ids": np.zeros(spatial, dtype=np.int32),
        "rescue_component_ids": np.zeros(spatial, dtype=np.int32),
        "core_target": np.zeros(spatial, dtype=np.bool_),
        "attached_halo": np.zeros(spatial, dtype=np.bool_),
        "detached_false_positive": np.zeros(spatial, dtype=np.bool_),
        "outer_ring": np.zeros(spatial, dtype=np.bool_),
        "halo_target": np.zeros(spatial, dtype=np.bool_),
        "far_background": np.ones(spatial, dtype=np.bool_),
        "baseline_rescue": np.zeros(spatial, dtype=np.bool_),
        "baseline_halo_advantage": np.zeros(spatial, dtype=np.bool_),
    }
    assert set(arrays) == set(cache.CACHE_ARRAY_KEYS)
    return arrays


def _synthetic_split_payload() -> tuple[dict[str, object], cache.TrainSplitAuthority]:
    official = ["sample_a", "sample_b", "sample_c"]
    development = ["sample_a", "sample_c"]
    internal = ["sample_b"]
    payload: dict[str, object] = {
        "schema": "synthetic_train_split/v1",
        "dataset": cache.DATASET_NAME,
        "source_split": "official_train_only",
        "split_seed": 17,
        "val_fraction": 1.0 / 3.0,
        "official_test_index_opened": False,
        "official_train_index_sha256": "a" * 64,
        "official_train_ids": official,
        "development_train_ids": development,
        "internal_validation_ids": internal,
        "mask_stats": {"synthetic": True},
        "data_protocol_manifest": {"synthetic": True},
    }
    payload["split_sha256"] = cache.canonical_sha256(payload)
    authority = cache.TrainSplitAuthority(
        schema="synthetic_train_split/v1",
        dataset=cache.DATASET_NAME,
        split_seed=17,
        validation_fraction=1.0 / 3.0,
        official_count=3,
        development_count=2,
        internal_count=1,
        canonical_split_sha256=str(payload["split_sha256"]),
        official_ids_sha256=cache.ordered_ids_sha256(official),
        development_ids_sha256=cache.ordered_ids_sha256(development),
        internal_ids_sha256=cache.ordered_ids_sha256(internal),
        official_train_index_sha256="a" * 64,
    )
    return payload, authority


def test_production_bindings_are_exact_and_zero_margin() -> None:
    assert cache.OFFICIAL_TRAIN_COUNT == 800
    assert cache.FULL_CONTEXT_HEIGHT == cache.FULL_CONTEXT_WIDTH == 512
    assert cache.CURRENT_TRAINING_STATE_KEYS == 568
    assert cache.CURRENT_TRAINING_STATE_TENSOR_MAPPING_SHA256 == (
        "d7600f61ee3d0967dae899de72a28f2e7e9e4c6381f2687189e45d84dcb3e298"
    )
    assert cache.CURRENT_INFERENCE_STATE_KEYS == 564
    assert cache.CURRENT_INFERENCE_STATE_SEMANTIC_SHA256 == (
        "f3745109e889cc6f25e42a43e698c5a43516ddc96a1364ffc78ab4b6b09d7f4f"
    )
    assert cache.BASELINE_TEACHER_EPOCH == 1000
    assert cache.OPERATIONAL_REFERENCE_EPOCH == 713
    assert cache.BASELINE_TEACHER_STATE_KEYS == 510
    assert str(cache.BASELINE_TEACHER_CHECKPOINT_PATH) == (
        "/home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_1000.pth.tar"
    )
    assert cache.BASELINE_TEACHER_CHECKPOINT_FILE_SHA256 == (
        "b4cb66be6e4a410dfd902ba050da82d0b666dd071bfb2c5477a7c3173ff07bc5"
    )
    assert cache.BASELINE_TEACHER_RAW_STATE_SEMANTIC_SHA256 == (
        "972e7c15f8da8142da85112f535fb555a86293e12d7341d7c5be653fb4076d9b"
    )
    assert cache.BASELINE_TEACHER_NORMALIZED_STATE_SEMANTIC_SHA256 == (
        "1961ed8ee278fde09508145fe537324172599bfa704c181dc53f756578070b5c"
    )
    assert cache.PERFORMANCE_MARGIN is None
    assert cache.PERFORMANCE_ACCEPTANCE_MARGIN is None
    assert all(value is False for value in cache.OFFICIAL_FALSE_FLAGS.values())
    assert cache.expected_determinism_manifest() == {
        "schema": "sctransnet_irstd_bgcr_cache_determinism_v1/v1",
        "seed": 42,
        "precision": "fp32",
        "default_dtype": "torch.float32",
        "model_and_cache_dtype": "torch.float32",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
        "autocast": False,
    }


def test_cache_module_has_no_dataset_or_evaluator_import_and_uses_formal_apis() -> None:
    source_path = Path(cache.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any("dataset" in name.lower() for name in imports)
    assert not any("evaluate" in name.lower() for name in imports)
    assert "build_formal_irstd_bgcr_model" in source
    assert "forward_for_irstd_training" in source
    assert "build_formal_teacher" in source
    assert "capture_outc_raw_logits" in source
    assert "torch.equal(torch.sigmoid(raw), probability)" in source
    assert "np.savez_compressed" not in source
    assert "np.savez(handle" in source
    assert '"container_compression": "store"' in source


def test_synthetic_train_split_validates_and_prohibited_access_fails_closed() -> None:
    payload, authority = _synthetic_split_payload()
    official, development, internal = cache.validate_train_split_payload(
        payload, authority=authority
    )
    assert official == ("sample_a", "sample_b", "sample_c")
    assert development == ("sample_a", "sample_c")
    assert internal == ("sample_b",)

    contaminated = dict(payload)
    contaminated["official_test_index_opened"] = True
    with pytest.raises(cache.IRSTDBGCRCacheError, match="prohibited"):
        cache.validate_train_split_payload(contaminated, authority=authority)

    overlap = dict(payload)
    overlap["internal_validation_ids"] = ["sample_a"]
    overlap["split_sha256"] = cache.canonical_sha256(
        {key: value for key, value in overlap.items() if key != "split_sha256"}
    )
    overlap_authority = replace(
        authority,
        canonical_split_sha256=str(overlap["split_sha256"]),
        internal_ids_sha256=cache.ordered_ids_sha256(["sample_a"]),
    )
    with pytest.raises(cache.IRSTDBGCRCacheError, match="partition"):
        cache.validate_train_split_payload(overlap, authority=overlap_authority)


def test_teacher_uses_semantic_state_hash_not_current_tensor_mapping_hash() -> None:
    normalized = {
        "weight": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    teacher_sha = cache.state_semantic_sha256(normalized)
    current_style_sha = cache.tensor_mapping_sha256(normalized)
    assert teacher_sha != current_style_sha
    raw = {f"model.{name}": value for name, value in normalized.items()}
    stripped = cache.normalize_baseline_teacher_state(
        raw,
        expected_keys=2,
        expected_state_sha256=teacher_sha,
    )
    assert tuple(stripped) == ("weight", "counter")
    with pytest.raises(cache.IRSTDBGCRCacheError, match="tensor-state SHA"):
        cache.normalize_baseline_teacher_state(
            raw,
            expected_keys=2,
            expected_state_sha256=current_style_sha,
        )


def test_train_mask_grayscale_is_binarized_by_frozen_strict_half_rule() -> None:
    loaded = cache.load_irstd_train_image_target(
        "XDU357",
        bound_ids=frozenset({"XDU357"}),
    )
    assert loaded.target.dtype == np.float32
    assert loaded.target.shape == (1, 512, 512)
    assert np.logical_or(loaded.target == 0.0, loaded.target == 1.0).all()
    assert int(loaded.target.sum()) == 150
    assert cache.TARGET_SOURCE_SCALE == 255.0
    assert cache.TARGET_BINARIZATION_THRESHOLD_SOURCE == 127.5
    assert cache.TARGET_BINARIZATION_THRESHOLD_NORMALIZED == 0.5
    assert cache.TARGET_BINARIZATION_COMPARISON == "strict_greater_than"


def test_epoch713_is_rejected_before_any_checkpoint_open(tmp_path: Path) -> None:
    nonexistent = tmp_path / "synthetic_epoch713.pth.tar"
    with pytest.raises(cache.IRSTDBGCRCacheError, match="epoch 1000"):
        cache._load_and_validate_teacher_checkpoint(  # noqa: SLF001
            nonexistent,
            expected_path=nonexistent,
            expected_file_sha256="0" * 64,
            expected_raw_state_semantic_sha256="1" * 64,
            expected_normalized_state_semantic_sha256="2" * 64,
            expected_state_keys=510,
            expected_epoch=713,
        )
    assert not nonexistent.exists()


def test_synthetic_cache_arrays_enforce_fp32_int32_bool_and_topology() -> None:
    arrays = _synthetic_arrays()
    cache.validate_cache_sample_arrays(arrays)
    contract = cache.cache_array_contract()
    assert contract["image"] == {"dtype": "float32", "shape": [1, 512, 512]}
    assert contract["u1"] == {"dtype": "float32", "shape": [32, 512, 512]}
    assert contract["target_component_ids"] == {
        "dtype": "int32",
        "shape": [512, 512],
    }
    assert contract["attached_halo"] == {"dtype": "bool", "shape": [512, 512]}

    bad_dtype = dict(arrays)
    bad_dtype["z_out"] = bad_dtype["z_out"].astype(np.float64)
    with pytest.raises(cache.IRSTDBGCRCacheError, match="dtype"):
        cache.validate_cache_sample_arrays(bad_dtype)

    bad_halo = dict(arrays)
    bad_halo["halo_target"] = bad_halo["halo_target"].copy()
    bad_halo["halo_target"][0, 0] = True
    with pytest.raises(cache.IRSTDBGCRCacheError, match="halo target"):
        cache.validate_cache_sample_arrays(bad_halo)


def test_atomic_sample_is_store_only_and_sidecar_binds_every_array(tmp_path: Path) -> None:
    staging = tmp_path / "synthetic.incomplete"
    (staging / "samples").mkdir(parents=True)
    (staging / "records").mkdir()
    arrays = _synthetic_arrays()
    source = {
        "image_relative_path": "IRSTD-1K/images/synthetic.png",
        "target_relative_path": "IRSTD-1K/masks/synthetic.png",
        "image_file_sha256": "1" * 64,
        "target_file_sha256": "2" * 64,
    }
    record = cache.write_cache_sample_atomic(
        staging,
        index=0,
        sample_id="synthetic",
        split_membership="development_train",
        arrays=arrays,
        source=source,
    )
    assert record["cache_relative_path"] == "samples/000000.npz"
    assert set(record["arrays"]) == set(cache.CACHE_ARRAY_KEYS)
    for flag, expected in cache.OFFICIAL_FALSE_FLAGS.items():
        assert record[flag] is expected
    assert len(record["cache_file_sha256"]) == 64
    with zipfile.ZipFile(staging / "samples/000000.npz", "r") as archive:
        assert archive.infolist()
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())


def test_resume_rejects_one_sided_sample_pair(tmp_path: Path) -> None:
    staging = tmp_path / "synthetic.incomplete"
    (staging / "samples").mkdir(parents=True)
    (staging / "records").mkdir()
    (staging / "samples/000000.npz").write_bytes(b"not-a-complete-pair")
    with pytest.raises(cache.IRSTDBGCRCacheError, match="one-sided"):
        cache._validate_sample_pair(  # noqa: SLF001 - intentional fail-closed test
            staging,
            index=0,
            sample_id="synthetic",
            split_membership="development_train",
        )


def test_manifest_contract_names_every_required_tensor_and_false_flag() -> None:
    assert cache.CACHE_ARRAY_KEYS == (
        "image",
        "target",
        "u1",
        "z_out",
        "z_d0",
        "z_gt2",
        "z_gt3",
        "z_gt4",
        "z_gt5",
        "baseline1000_logits",
        "target_component_ids",
        "rescue_component_ids",
        "core_target",
        "attached_halo",
        "detached_false_positive",
        "outer_ring",
        "halo_target",
        "far_background",
        "baseline_rescue",
        "baseline_halo_advantage",
    )
    assert set(cache.OFFICIAL_FALSE_FLAGS) == {
        "official_test_accessed",
        "official_test_index_opened",
        "official_test_index_parsed",
        "official_test_loader_built",
        "official_evaluation_performed",
    }
