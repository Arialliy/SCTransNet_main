from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis import analyze_three_dataset_qfg_level_knockout_v1 as analyzer


class _FakeLevel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))


@dataclass(frozen=True)
class _PreparedLevel:
    gate: torch.Tensor
    factor: torch.Tensor


@dataclass(frozen=True)
class _Prepared:
    levels: tuple[_PreparedLevel, ...]


class _FakeQFG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [_FakeLevel(value) for value in (0.2, 0.3, 0.4, 0.5)]
        )

    def prepare(self, features, query_sizes):
        del query_sizes
        prepared = []
        for index, (level, feature) in enumerate(zip(self.levels, features)):
            gate = (0.15 + 0.02 * index) * torch.tanh(feature[:, :1])
            factor = 1.0 + torch.tanh(level.alpha) * gate
            prepared.append(_PreparedLevel(gate=gate, factor=factor))
        return _Prepared(levels=tuple(prepared))

    def apply_prepared(self, queries, prepared):
        outputs = tuple(
            query * level.factor
            for query, level in zip(tuple(queries), prepared.levels)
        )
        return SimpleNamespace(
            queries=outputs,
            factors=tuple(level.factor for level in prepared.levels),
            gates=tuple(level.gate for level in prepared.levels),
        )


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tpd_qfg = _FakeQFG()

    def forward(self, value):
        features = tuple(value for _ in range(4))
        prepared = self.tpd_qfg.prepare(
            features, tuple(tuple(value.shape[-2:]) for _ in range(4))
        )
        queries = tuple(value * float(index + 1) for index in range(4))
        gated = self.tpd_qfg.apply_prepared(queries, prepared)
        combined = sum(gated.queries) / 10.0
        return torch.sigmoid(combined)


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = (
            (
                torch.tensor([[[-2.0, 2.0], [-1.0, 1.0]]]),
                torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
                "sample_a",
            ),
            (
                torch.tensor([[[2.0, -2.0], [1.5, -1.5]]]),
                torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                "sample_b",
            ),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image, mask, sample_id = self.samples[index]
        return image, mask, (2, 2), sample_id


def test_public_zero_based_modes_map_to_exact_existing_primitives() -> None:
    expected = {
        "full": ("full", []),
        "level0_off": ("level_1_off", [0]),
        "level1_off": ("level_2_off", [1]),
        "level2_off": ("level_3_off", [2]),
        "level3_off": ("level_4_off", [3]),
        "all_off": ("all_off", [0, 1, 2, 3]),
    }
    assert tuple(expected) == analyzer.PUBLIC_MODES
    for public_mode, (primitive, selected) in expected.items():
        binding = analyzer.normalize_public_mode(public_mode)
        assert binding["primitive_mode"] == primitive
        assert binding["knockout_level_indices_zero_based"] == selected


def test_probability_difference_uses_global_unpadded_pixel_denominator() -> None:
    full = [
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.5], [0.25]], dtype=np.float32),
    ]
    other = [
        np.array([[0.1, 0.8]], dtype=np.float32),
        np.array([[0.5], [0.15]], dtype=np.float32),
    ]
    result = analyzer.probability_difference(full, other)
    expected_sum = (
        abs(float(np.float32(0.0)) - float(np.float32(0.1)))
        + abs(float(np.float32(1.0)) - float(np.float32(0.8)))
        + abs(float(np.float32(0.5)) - float(np.float32(0.5)))
        + abs(float(np.float32(0.25)) - float(np.float32(0.15)))
    )
    assert result["scope"] == "all_original_unpadded_test_pixels"
    assert result["element_count"] == 4
    assert result["absolute_difference_sum"] == pytest.approx(expected_sum)
    assert result["mean_abs"] == pytest.approx(expected_sum / 4.0)
    assert result["max_abs"] == pytest.approx(0.2)
    assert result["functionally_different"] is True


def test_cpu_synthetic_six_mode_audit_is_exact_and_restores_state() -> None:
    model = _FakeModel().eval()
    loader = DataLoader(_TinyDataset(), batch_size=1, shuffle=False, num_workers=0)
    state_before = analyzer.stage2_audit.module_state_sha256(model)
    alpha_before = analyzer.qfg_audit.alpha_state_sha256(model)

    result = analyzer.analyze_loaded_model(
        model,
        loader,
        torch.device("cpu"),
        ("sample_a", "sample_b"),
        sweep_thresholds=(0.0, 0.5, 1.0),
    )

    assert tuple(result["modes"]) == analyzer.PUBLIC_MODES
    assert result["restoration_audit"]["model_state_unchanged"] is True
    assert result["restoration_audit"]["alpha_state_unchanged"] is True
    assert analyzer.stage2_audit.module_state_sha256(model) == state_before
    assert analyzer.qfg_audit.alpha_state_sha256(model) == alpha_before
    assert "prepare" not in model.tpd_qfg.__dict__
    assert "apply_prepared" not in model.tpd_qfg.__dict__

    full = result["modes"]["full"]
    assert full["probability_difference_to_full"]["max_abs"] == 0.0
    assert full["probability_difference_to_full"]["mean_abs"] == 0.0
    assert full["probability_difference_to_full"]["equivalent"] is True
    assert full["fixed_threshold_0_5"]["threshold"] == 0.5
    assert isinstance(full["fixed_threshold_0_5"]["false_positive_pixels"], int)
    assert "unmatched_predicted_pixels" in full["fixed_threshold_0_5"]
    assert full["query_perturbation"]["levels"][0][
        "query_perturbation_rms"
    ] > 0.0
    assert (
        full["spatial_gate_factor_statistics"]["levels"][0][
            "level_index_zero_based"
        ]
        == 0
    )
    assert "target_minus_hard_negative" in (
        full["spatial_gate_factor_statistics"]["levels"][0]["gate_contrasts"]
    )

    level0 = result["modes"]["level0_off"]
    assert level0["alpha_knockout"]["selected_level_indices_zero_based"] == [0]
    assert level0["query_perturbation"]["levels"][0][
        "query_perturbation_rms"
    ] == 0.0
    assert level0["query_perturbation"]["levels"][1][
        "query_perturbation_rms"
    ] > 0.0
    all_off = result["modes"]["all_off"]
    assert all_off["alpha_knockout"]["selected_level_indices_zero_based"] == [
        0,
        1,
        2,
        3,
    ]
    assert all(
        level["query_perturbation_rms"] == 0.0
        and level["factor_minus_one_rms"] == 0.0
        for level in all_off["query_perturbation"]["levels"]
    )
    assert all_off["probability_difference_to_full"]["element_count"] == 8
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
        **result,
        "reference_replay_audit": {"passed": True},
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
    }
    analyzer.validate_output_payload(payload)


def test_reference_replay_requires_exact_counts_and_frozen_tolerances() -> None:
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
    audit = analyzer.reference_replay_audit(observed, reference)
    assert audit["passed"] is True
    assert audit["compared"]["miou"]["absolute_tolerance"] == 1e-4

    wrong = dict(observed)
    wrong["unmatched_predicted_pixels"] += 1
    with pytest.raises(
        ValueError, match="reference replay count differs: unmatched_predicted_pixels"
    ):
        analyzer.reference_replay_audit(wrong, reference)


def test_atomic_json_is_true_write_once(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    analyzer.atomic_create_json(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError, match="refusing existing output"):
        analyzer.atomic_create_json(output, {"value": 2})


def test_module_contract_imports_under_python_optimized_mode() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    command = (
        "from analysis import analyze_three_dataset_qfg_level_knockout_v1 as a; "
        "b=a.normalize_public_mode('level3_off'); "
        "\nif b['knockout_level_indices_zero_based'] != [3]: raise SystemExit(7); "
        "\nif a.CHECKPOINT_ROLE != 'best_miou': raise SystemExit(8)"
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
