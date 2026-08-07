"""Static and synthetic contracts for the cache-only IRSTD-BGCR trainer.

These tests deliberately use no dataset loader, split index, official-test
artifact, or production checkpoint.  They exercise only deterministic plans,
RAM-resident cache reads, local repair-head crops, and artifact parsers.
"""

from __future__ import annotations

import ast
from collections import Counter
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments import cache_irstd_frozen_context_v1 as cache_contract
from experiments import irstd_bgcr_run_contract as run_contract
from experiments import train_irstd_bgcr_v1 as trainer
from model.irstd_core_ring_repair import (
    IRSTDCoreRingRepairHead,
    LocalGroupNorm2d,
    PRODUCTION_PARAMETER_COUNT,
)


def test_trainer_project_imports_are_cache_model_and_metric_only() -> None:
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    observed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            observed.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module in ("experiments", "model"):
                observed.update(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                observed.add(node.module)
    project_imports = {
        name
        for name in observed
        if name.startswith("experiments.") or name.startswith("model.")
    }
    assert project_imports == {
        "experiments.cache_irstd_frozen_context_v1",
        "experiments.irstd_bgcr_run_contract",
        "experiments.irstd_core_ring_loss",
        "experiments.irstd_logit_counterfactual",
        "experiments.pbdr_v4_metric_core",
        "experiments.pbdr_v4_models_seed42_v1",
        "experiments.pbdr_v4_run_artifacts",
        "experiments.pbdr_v4_state_contract",
        "experiments.pbdr_v4_training_core",
        "model.irstd_core_ring_repair",
        "model.tpd8_ner4_qfg2_irstd_crr",
    }


def test_cache_array_contract_is_exactly_the_committed_cache_contract() -> None:
    contract = cache_contract.cache_array_contract()
    assert tuple(trainer.CACHE_FIELDS) == tuple(cache_contract.CACHE_ARRAY_KEYS)
    assert set(contract) == set(trainer.CACHE_FIELDS)
    assert contract["image"] == {"dtype": "float32", "shape": [1, 512, 512]}
    assert contract["target"] == {"dtype": "float32", "shape": [1, 512, 512]}
    assert contract["u1"] == {"dtype": "float32", "shape": [32, 512, 512]}
    for name in trainer.CACHE_ID_FIELDS:
        assert contract[name] == {"dtype": "int32", "shape": [512, 512]}
    for name in trainer.CACHE_MASK_FIELDS:
        assert contract[name] == {"dtype": "bool", "shape": [512, 512]}


def _balanced_support() -> tuple[tuple[str, ...], dict[str, tuple[bool, bool]]]:
    identifiers = tuple(f"sample-{index:03d}" for index in range(48))
    support: dict[str, tuple[bool, bool]] = {}
    for index, sample_id in enumerate(identifiers):
        support[sample_id] = (
            (True, False)
            if index < 16
            else (False, True)
            if index < 32
            else (False, False)
        )
    return identifiers, support


def test_epoch_plan_is_stable_balanced_and_uses_each_sample_once() -> None:
    identifiers, support = _balanced_support()
    first = trainer.build_epoch_plan(identifiers, support, epoch=17)
    replay = trainer.build_epoch_plan(identifiers, support, epoch=17)
    next_epoch = trainer.build_epoch_plan(identifiers, support, epoch=18)
    assert first == replay
    assert trainer.epoch_plan_sha256(first) == trainer.epoch_plan_sha256(replay)
    assert first != next_epoch
    assert len(first) == len({item.sample_id for item in first}) == 48
    assert {item.sample_id for item in first} == set(identifiers)
    for start in range(0, len(first), trainer.BATCH_SIZE):
        batch = first[start : start + trainer.BATCH_SIZE]
        source_counts = Counter(item.source_class for item in batch)
        mode_counts = Counter(item.counterfactual_mode for item in batch)
        assert set(source_counts) == set(trainer.SOURCE_CLASSES)
        assert set(mode_counts) == set(trainer.COUNTERFACTUAL_MODES)
        assert max(source_counts.values()) - min(source_counts.values()) <= 1
        assert max(mode_counts.values()) - min(mode_counts.values()) <= 1


def test_epoch_plan_fails_closed_without_error_aware_support() -> None:
    identifiers = tuple(f"sample-{index:03d}" for index in range(48))
    support = {sample_id: (False, False) for sample_id in identifiers}
    with pytest.raises(trainer.IRSTDBGCRTrainingError, match="insufficient core"):
        trainer.build_epoch_plan(identifiers, support, epoch=1)


def test_selected_and_random_centers_keep_the_256_loss_inside_512() -> None:
    low = np.zeros((512, 512), dtype=np.bool_)
    low[0, 0] = True
    high = np.zeros((512, 512), dtype=np.bool_)
    high[-1, -1] = True
    assert trainer._coordinate_from_mask(
        low,
        epoch=1,
        tag="low",
        sample_id="sample-low",
    ) == (128, 128)
    assert trainer._coordinate_from_mask(
        high,
        epoch=1,
        tag="high",
        sample_id="sample-high",
    ) == (384, 384)
    for index in range(100):
        y, x = trainer._random_coordinate(epoch=7, sample_id=f"sample-{index}")
        assert 128 <= y <= 384
        assert 128 <= x <= 384


def test_boundary_outer_crop_has_only_eight_pixel_zero_halo() -> None:
    source = np.arange(512 * 512, dtype=np.float32).reshape(1, 512, 512)
    low = trainer.extract_outer_context_patch(source, center_y=128, center_x=128)
    high = trainer.extract_outer_context_patch(source, center_y=384, center_x=384)
    assert low.shape == high.shape == (1, 272, 272)
    assert np.count_nonzero(low[:, :8, :]) == 0
    assert np.count_nonzero(low[:, :, :8]) == 0
    assert np.count_nonzero(high[:, -8:, :]) == 0
    assert np.count_nonzero(high[:, :, -8:]) == 0
    assert np.array_equal(low[:, 8:264, 8:264], source[:, :256, :256])
    assert np.array_equal(high[:, 8:264, 8:264], source[:, 256:, 256:])


def _head_inputs(height: int, width: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260808)
    one_channel = {
        name: torch.randn(1, 1, height, width, generator=generator)
        for name in ("image", "z_out", "z_d0", "z_gt2", "z_gt3", "z_gt4", "z_gt5")
    }
    one_channel["local_feature"] = torch.randn(
        1,
        32,
        height,
        width,
        generator=generator,
    )
    return one_channel


def test_272_style_halo_makes_center_head_exactly_equal_to_full_head() -> None:
    torch.manual_seed(42)
    head = IRSTDCoreRingRepairHead().eval()
    assert head.architecture_manifest()["maximum_spatial_receptive_radius"] == 8
    assert any(isinstance(module, LocalGroupNorm2d) for module in head.modules())
    assert not any(isinstance(module, nn.GroupNorm) for module in head.modules())
    with torch.no_grad():
        head.positive_residual_head.weight.fill_(0.015625)
        head.positive_residual_head.bias.fill_(0.03125)
        head.negative_residual_head.weight.fill_(-0.0078125)
        head.negative_residual_head.bias.fill_(0.015625)
        full_inputs = _head_inputs(48, 48)
        full = head.forward_with_diagnostics(**full_inputs)
        crop_inputs = {
            name: value[..., 8:40, 8:40]
            for name, value in full_inputs.items()
        }
        crop = head.forward_with_diagnostics(**crop_inputs)
    for name in (
        "routed_logits",
        "delta_logits",
        "positive_delta",
        "negative_delta",
        "core_gate_logits",
        "halo_gate_logits",
        "core_gate",
        "halo_gate",
    ):
        torch.testing.assert_close(
            getattr(crop, name)[..., 8:24, 8:24],
            getattr(full, name)[..., 16:32, 16:32],
            rtol=0.0,
            atol=0.0,
        )


def test_resident_cache_load_never_reopens_npz() -> None:
    cache = object.__new__(trainer.FrozenContextCache)
    sample = trainer.FrozenCacheSample(
        sample_id="sample-000",
        arrays={"target": np.zeros((1, 512, 512), dtype=np.float32)},
    )
    cache._samples = {sample.sample_id: sample}
    with patch.object(np, "load", side_effect=AssertionError("disk reopen")):
        assert cache.load(sample.sample_id) is sample
        assert cache.load(sample.sample_id) is sample


class _HeadOnlyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.irstd_repair = IRSTDCoreRingRepairHead()

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.irstd_repair.parameters())


def test_optimizer_scheduler_and_precision_recipe_are_frozen() -> None:
    model = _HeadOnlyModel()
    optimizer = trainer.build_optimizer(model)  # type: ignore[arg-type]
    scheduler = trainer.build_scheduler(optimizer)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "irstd_repair"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1.0e-4)
    assert len(optimizer.param_groups[0]["params"]) == 29
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        PRODUCTION_PARAMETER_COUNT
    )
    assert scheduler.T_max == 120
    assert scheduler.eta_min == pytest.approx(1.0e-6)
    recipe = trainer._training_recipe()
    assert recipe["precision"] == "fp32"
    assert recipe["tf32"] is False
    assert recipe["seed"] == 42
    assert recipe["batch_size"] == 16
    assert recipe["host_resident_cache"] is True
    for flag, expected in run_contract.OFFICIAL_FALSE_FLAGS.items():
        assert recipe[flag] is expected
    run_source = inspect.getsource(trainer.run)
    assert "torch.set_default_dtype(torch.float32)" in run_source
    assert 'torch.set_float32_matmul_precision("highest")' in run_source


def _synthetic_outer_selection() -> dict[str, object]:
    inner: dict[str, object] = {
        "schema": run_contract.SELECTION_SCHEMA,
        "dataset": trainer.DATASET,
        "role": trainer.ROLE,
        "candidate_epochs": list(run_contract.OOF_EVALUATION_EPOCHS),
        "selected_epoch": 5,
        "fold_assignment_sha256": run_contract.FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": (
            run_contract.SOURCE_SPLIT_MANIFEST_FILE_SHA256
        ),
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }
    outer: dict[str, object] = {
        "schema": trainer.OOF_SELECTOR_SCHEMA,
        "dataset": trainer.DATASET,
        "role": trainer.ROLE,
        "source_scope": run_contract.SOURCE_SCOPE,
        "selection": inner,
        "selected_epoch": 5,
        "fold_summaries": [
            {
                "fold_index": fold,
                "path": f"/synthetic/fold-{fold}.json",
                "bytes": fold + 1,
                "file_sha256": f"{fold + 1:x}" * 64,
            }
            for fold in run_contract.FOLD_TIE_ORDER
        ],
        "fold_assignment_sha256": run_contract.FOLD_ASSIGNMENT_SHA256,
        "fold_manifest_sha256": "d" * 64,
        "source_split_manifest_file_sha256": (
            run_contract.SOURCE_SPLIT_MANIFEST_FILE_SHA256
        ),
        "probability_threshold": run_contract.PROBABILITY_THRESHOLD,
        "probability_comparison": run_contract.PROBABILITY_COMPARISON,
        "performance_acceptance_margin": None,
        **run_contract.OFFICIAL_FALSE_FLAGS,
    }
    outer["selection_sha256"] = trainer.canonical_json_sha256(outer)
    return outer


def test_full_mode_accepts_the_selector_outer_envelope_and_rejects_drift(
    tmp_path: Path,
) -> None:
    payload = _synthetic_outer_selection()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed, epoch = trainer._load_oof_selection(path)
    assert observed == payload
    assert epoch == 5

    payload["source_scope"] = "wrong"
    payload["selection_sha256"] = trainer.canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "selection_sha256"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(trainer.IRSTDBGCRTrainingError, match="source scope"):
        trainer._load_oof_selection(path)


def test_orphan_training_sidecars_fail_closed(tmp_path: Path) -> None:
    trainer._require_no_orphan_training_sidecars(tmp_path)
    (tmp_path / trainer.METRIC_HISTORY_NAME).write_text("{}\n", encoding="utf-8")
    with pytest.raises(trainer.IRSTDBGCRTrainingError, match="orphan"):
        trainer._require_no_orphan_training_sidecars(tmp_path)


def test_cli_exposes_only_frozen_cache_fold_and_resume_inputs() -> None:
    parser = trainer.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "fold",
            "--fold-index",
            "0",
            "--cache-root",
            "/cache",
            "--fold-manifest",
            "/folds.json",
            "--run-dir",
            "/run",
            "--device",
            "cuda:1",
        ]
    )
    assert args.mode == "fold"
    assert args.fold_index == 0
    assert args.cache_root == Path("/cache")
    assert args.fold_manifest == Path("/folds.json")
    assert args.oof_selection is None
    assert args.resume == "auto"
    assert args.device == "cuda:1"
    assert not hasattr(args, "dataset_root")
    assert not hasattr(args, "official_test_index")
