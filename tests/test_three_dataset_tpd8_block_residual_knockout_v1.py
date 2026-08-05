from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis import (
    analyze_three_dataset_tpd8_block_residual_knockout_v1 as analyzer,
)
from model.tpd_clean_v8_mprs_dch import (
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    build_clean_v8_mprs_dch_patch_embedding,
)


class _FakeMTC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings_1 = build_clean_v8_mprs_dch_patch_embedding(
            PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            channels=2,
            stride=16,
        )
        self.embeddings_2 = build_clean_v8_mprs_dch_patch_embedding(
            PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            channels=4,
            stride=8,
        )


class _FakeTPD8Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(731)
            self.stem1 = nn.Conv2d(1, 2, kernel_size=1)
            self.stem2 = nn.Conv2d(1, 4, kernel_size=1)
            self.mtc = _FakeMTC()
        with torch.no_grad():
            for index, path in enumerate(analyzer.BLOCK_PATHS):
                block = self.get_submodule(path)
                block.saliency_scale.fill_(0.18 + 0.025 * index)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.stem1(value))
        x2 = F.relu(self.stem2(F.avg_pool2d(value, kernel_size=2, stride=2)))
        emb1, evidence1 = self.mtc.embeddings_1.forward_with_evidence(x1)
        emb2, evidence2 = self.mtc.embeddings_2.forward_with_evidence(x2)
        assert emb1 is not None and emb2 is not None
        evidence = (*evidence1, *evidence2)
        logit = 0.55 * value
        logit = logit + 0.20 * F.interpolate(
            emb1.mean(dim=1, keepdim=True),
            size=value.shape[-2:],
            mode="nearest",
        )
        logit = logit + 0.15 * F.interpolate(
            emb2.mean(dim=1, keepdim=True),
            size=value.shape[-2:],
            mode="nearest",
        )
        for state in evidence:
            logit = logit + 0.025 * F.interpolate(
                state.mean(dim=1, keepdim=True),
                size=value.shape[-2:],
                mode="nearest",
            )
        return torch.sigmoid(logit)


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        axis = torch.linspace(-2.0, 2.0, 32)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        image_a = (0.7 * grid_x + 0.3 * grid_y).unsqueeze(0)
        image_b = (0.4 * grid_x - 0.8 * grid_y).flip(-1).unsqueeze(0)
        image_a[:, 8:10, 11:13] += 3.0
        image_b[:, 19:22, 7:9] += 2.5
        mask_a = torch.zeros((1, 32, 32), dtype=torch.float32)
        mask_b = torch.zeros((1, 32, 32), dtype=torch.float32)
        mask_a[:, 8:10, 11:13] = 1.0
        mask_b[:, 19:22, 7:9] = 1.0
        self.samples = (
            (image_a, mask_a, (31, 29), "sample_a"),
            (image_b, mask_b, (30, 32), "sample_b"),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _model_and_loader() -> tuple[_FakeTPD8Model, DataLoader]:
    model = _FakeTPD8Model().eval()
    loader = DataLoader(_TinyDataset(), batch_size=1, shuffle=False, num_workers=0)
    return model, loader


def test_mode_order_and_exact_block_bindings() -> None:
    expected = {
        "full": [],
        "e1b0_off": [analyzer.BLOCK_PATHS[0]],
        "e1b1_off": [analyzer.BLOCK_PATHS[1]],
        "e1b2_off": [analyzer.BLOCK_PATHS[2]],
        "e1b3_off": [analyzer.BLOCK_PATHS[3]],
        "e2b0_off": [analyzer.BLOCK_PATHS[4]],
        "e2b1_off": [analyzer.BLOCK_PATHS[5]],
        "e2b2_off": [analyzer.BLOCK_PATHS[6]],
        "all7_off": list(analyzer.BLOCK_PATHS),
    }
    assert tuple(expected) == analyzer.PUBLIC_MODES
    for mode, paths in expected.items():
        binding = analyzer.normalize_public_mode(mode)
        assert binding["knockout_block_paths"] == paths
        assert binding["knockout_block_indices_zero_based"] == [
            analyzer.BLOCK_PATHS.index(path) for path in paths
        ]


def test_saliency_scale_knockout_is_valuewise_and_exception_safe() -> None:
    model, _ = _model_and_loader()
    source_scale_sha = analyzer.saliency_scale_state_sha256(model)
    source_model_sha = analyzer.stage2_audit.module_state_sha256(model)
    snapshots = {
        path: model.get_submodule(path).saliency_scale.detach().clone()
        for path in analyzer.BLOCK_PATHS
    }

    with pytest.raises(RuntimeError, match="intentional"):
        with analyzer.temporary_saliency_scale_knockout(
            model, "e1b2_off"
        ) as audit:
            assert torch.count_nonzero(
                model.get_submodule(analyzer.BLOCK_PATHS[2]).saliency_scale
            ).item() == 0
            for index, path in enumerate(analyzer.BLOCK_PATHS):
                if index != 2:
                    assert torch.equal(
                        model.get_submodule(path).saliency_scale, snapshots[path]
                    )
            assert audit["source_saliency_scale_sha256"] == source_scale_sha
            assert audit["active_saliency_scale_sha256"] != source_scale_sha
            assert audit["active_model_state_sha256"] != source_model_sha
            raise RuntimeError("intentional")

    assert analyzer.saliency_scale_state_sha256(model) == source_scale_sha
    assert analyzer.stage2_audit.module_state_sha256(model) == source_model_sha
    for path in analyzer.BLOCK_PATHS:
        assert torch.equal(model.get_submodule(path).saliency_scale, snapshots[path])


def test_full_statistics_wrapper_returns_bitwise_original_and_restores_lookup() -> None:
    model, _ = _model_and_loader()
    samples = [_TinyDataset()[0], _TinyDataset()[1]]
    state_before = analyzer.stage2_audit.module_state_sha256(model)
    with torch.inference_mode():
        expected = [model(image.unsqueeze(0)) for image, _, _, _ in samples]
        with analyzer.capture_full_mprs_statistics(model) as recorder:
            observed = []
            for image, mask, size, _ in samples:
                recorder.begin_batch(mask.unsqueeze(0), size)
                observed.append(model(image.unsqueeze(0)))
                recorder.end_batch()
        summary = recorder.summary()

    assert all(torch.equal(left, right) for left, right in zip(observed, expected))
    assert summary["production_output_policy"] == (
        "return_original_forward_output_unchanged"
    )
    assert summary["temporary_forward_wrappers_restored"] is True
    assert summary["block_order"] == list(analyzer.BLOCK_PATHS)
    assert len(summary["blocks"]) == 7
    strict_padding_exclusion_count = 0
    for row in summary["blocks"]:
        assert row["forward_call_count"] == 2
        statistics = row["rms_statistics"]
        assert set(statistics) == set(analyzer._TERM_NAMES)
        assert statistics["keep_K"]["rms"] is not None
        assert statistics["residual_R"]["rms"] > 0.0
        assert statistics["target_residual_R"]["element_count"] > 0
        assert statistics["background_residual_R"]["element_count"] > 0
        covered = (
            statistics["target_residual_R"]["element_count"]
            + statistics["background_residual_R"]["element_count"]
        )
        assert covered <= statistics["residual_R"]["element_count"]
        strict_padding_exclusion_count += int(
            covered < statistics["residual_R"]["element_count"]
        )
        assert row["target_minus_background_residual_rms"] == pytest.approx(
            statistics["target_residual_R"]["rms"]
            - statistics["background_residual_R"]["rms"]
        )
    assert strict_padding_exclusion_count >= 1
    assert analyzer.stage2_audit.module_state_sha256(model) == state_before
    assert all(
        "forward" not in model.get_submodule(path).__dict__
        and "aligned_mprs_terms" not in model.get_submodule(path).__dict__
        for path in analyzer.BLOCK_PATHS
    )


def test_full_statistics_wrapper_clears_active_batch_and_restores_on_exception() -> None:
    model, _ = _model_and_loader()
    image, mask, _, _ = _TinyDataset()[0]
    state_before = analyzer.stage2_audit.module_state_sha256(model)
    with pytest.raises(RuntimeError, match="after-forward failure"):
        with torch.inference_mode():
            with analyzer.capture_full_mprs_statistics(model) as recorder:
                recorder.begin_batch(mask.unsqueeze(0), (31, 29))
                model(image.unsqueeze(0))
                raise RuntimeError("after-forward failure")
    assert recorder.current_target is None
    assert recorder.aborted_batch_count == 1
    assert recorder.hooks_restored is True
    assert analyzer.stage2_audit.module_state_sha256(model) == state_before
    assert all(
        "forward" not in model.get_submodule(path).__dict__
        and "aligned_mprs_terms" not in model.get_submodule(path).__dict__
        for path in analyzer.BLOCK_PATHS
    )


def test_probability_difference_uses_every_original_pixel() -> None:
    full = [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.5], [0.25]], dtype=np.float32),
    ]
    other = [
        np.array([[0.1, 0.8]], dtype=np.float32),
        np.array([[0.5], [0.15]], dtype=np.float32),
    ]
    result = analyzer.probability_difference(full, other)
    assert result["scope"] == "all_original_unpadded_test_pixels"
    assert result["element_count"] == 4
    assert result["absolute_difference_sum"] == pytest.approx(0.4)
    assert result["mean_abs"] == pytest.approx(0.1)
    assert result["max_abs"] == pytest.approx(0.2)


def test_cpu_synthetic_nine_mode_audit_restores_state_and_uses_two_thresholds() -> None:
    model, loader = _model_and_loader()
    state_before = analyzer.stage2_audit.module_state_sha256(model)
    scale_before = analyzer.saliency_scale_state_sha256(model)
    result = analyzer.analyze_loaded_model(
        model,
        loader,
        torch.device("cpu"),
        ("sample_a", "sample_b"),
    )

    assert tuple(result["modes"]) == analyzer.PUBLIC_MODES
    assert result["restoration_audit"]["model_state_unchanged"] is True
    assert result["restoration_audit"]["saliency_scale_unchanged"] is True
    assert analyzer.stage2_audit.module_state_sha256(model) == state_before
    assert analyzer.saliency_scale_state_sha256(model) == scale_before
    assert all(
        "forward" not in model.get_submodule(path).__dict__
        and "aligned_mprs_terms" not in model.get_submodule(path).__dict__
        for path in analyzer.BLOCK_PATHS
    )

    full = result["modes"]["full"]
    assert full["probability_difference_to_full"]["max_abs"] == 0.0
    assert full["probability_difference_to_full"]["mean_abs"] == 0.0
    assert full["sweep_thresholds"] == [0.5, 1.0]
    assert full["fixed_threshold_0_5"]["threshold"] == 0.5
    assert full["full_mprs_statistics"]["block_count"] == 7
    assert full["saliency_scale_knockout"][
        "source_saliency_scale_sha256"
    ] == full["saliency_scale_knockout"]["active_saliency_scale_sha256"]

    for mode_name, mode in result["modes"].items():
        difference = mode["probability_difference_to_full"]
        assert "absolute_difference_sum" in difference
        assert difference["element_count"] == mode["fixed_threshold_0_5"][
            "valid_pixel_count"
        ]
        assert difference["mean_abs"] == (
            difference["absolute_difference_sum"] / difference["element_count"]
        )
        points = mode["descriptive_pd_fa"]["points"]
        assert [point["threshold"] for point in points] == [0.5, 1.0]
        assert points[1]["selected_point_is_empty"] is True
        assert points[1]["pd"] == 0.0
        assert points[1]["fa"] == 0.0
        knockout = mode["saliency_scale_knockout"]
        assert knockout["restored_saliency_scale_sha256"] == knockout[
            "source_saliency_scale_sha256"
        ]
        assert knockout["restored_model_state_sha256"] == knockout[
            "source_model_state_sha256"
        ]
        assert knockout["selected_active_nonzero_count"] == 0
        assert all(
            record["active_nonzero_count"] == 0
            for record in knockout["selected_vectors"]
        )
        if mode_name != "full" and knockout["selected_source_nonzero_count"] > 0:
            assert knockout["active_saliency_scale_sha256"] != knockout[
                "source_saliency_scale_sha256"
            ]

    one_off = result["modes"]["e2b1_off"]
    assert one_off["saliency_scale_knockout"][
        "selected_block_paths"
    ] == [analyzer.BLOCK_PATHS[5]]
    assert one_off["full_mprs_statistics"] is None
    assert one_off["saliency_scale_knockout"][
        "active_saliency_scale_sha256"
    ] != scale_before
    all_off = result["modes"]["all7_off"]
    assert all_off["knockout_block_paths"] == list(analyzer.BLOCK_PATHS)
    assert all_off["probability_difference_to_full"]["element_count"] == (
        31 * 29 + 30 * 32
    )
    assert all_off["probability_difference_to_full"]["functionally_different"]
    assert result["probability_arrays_persisted"] is False

    payload = {
        "schema": analyzer.SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "checkpoint_role": analyzer.CHECKPOINT_ROLE,
        "seed": 42,
        "test_selected": True,
        "evaluation_protocol": analyzer.EVALUATION_PROTOCOL,
        "sweep_thresholds": list(analyzer.SWEEP_THRESHOLDS),
        "mode_order": list(analyzer.PUBLIC_MODES),
        **result,
        "reference_replay_audit": {"passed": True},
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
    }
    analyzer.validate_output_payload(payload)
    analyzer.validate_output_payload(json.loads(json.dumps(payload, allow_nan=False)))


def test_reference_replay_and_atomic_write_reuse_frozen_skeleton(tmp_path: Path) -> None:
    reference = {
        "threshold": 0.5,
        "miou": 0.8,
        "niou": 0.7,
        "pd": 0.9,
        "fa": 1e-5,
        "pixel_precision": 0.85,
        "pixel_recall": 0.88,
        "pixel_f1": 0.865,
        "test_loss": 0.001,
        "target_count": 10,
        "matched_target_count": 9,
        "tiny_target_count": 3,
        "matched_tiny_target_count": 2,
        "predicted_object_count": 11,
        "unmatched_predicted_object_count": 2,
        "unmatched_predicted_pixels": 7,
        "valid_pixel_count": 1000,
    }
    observed = dict(reference)
    observed["miou"] += 5e-5
    assert analyzer.reference_replay_audit(observed, reference)["passed"] is True
    wrong = dict(observed)
    wrong["unmatched_predicted_pixels"] += 1
    with pytest.raises(ValueError, match="reference replay count differs"):
        analyzer.reference_replay_audit(wrong, reference)

    output = tmp_path / "result.json"
    analyzer.atomic_create_json(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError, match="refusing existing output"):
        analyzer.atomic_create_json(output, {"value": 2})


def test_module_contract_imports_under_python_optimized_mode() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = (
        "from analysis import "
        "analyze_three_dataset_tpd8_block_residual_knockout_v1 as a; "
        "b=a.normalize_public_mode('e2b2_off'); "
        "\nif b['knockout_block_paths'] != [a.BLOCK_PATHS[6]]: raise SystemExit(7); "
        "\nif a.PUBLIC_MODES[-1] != 'all7_off': raise SystemExit(8); "
        "\nif a.SWEEP_THRESHOLDS != (0.5, 1.0): raise SystemExit(9)"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", command],
        cwd=analyzer.REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
