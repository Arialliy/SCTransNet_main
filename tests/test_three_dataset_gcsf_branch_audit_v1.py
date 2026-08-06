from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from analysis import analyze_three_dataset_gcsf_branch_audit_v1 as subject


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        image1 = torch.linspace(-1.0, 1.0, 64).reshape(1, 8, 8)
        image2 = torch.flip(image1, dims=(-1,)) + 0.2
        mask1 = torch.zeros_like(image1)
        mask2 = torch.zeros_like(image2)
        mask1[:, 2:4, 3:5] = 1.0
        mask2[:, 4:7, 1:3] = 1.0
        self.samples = (
            (image1, mask1, (7, 6), "sample_1"),
            (image2, mask2, (6, 8), "sample_2"),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _fake_preparer(counter: dict[str, int]):
    def prepare(model: nn.Module, images: torch.Tensor) -> subject.ForwardLocalBranches:
        counter["prepare"] += 1
        encoder = (
            images.repeat(1, 2, 1, 1),
            F.avg_pool2d(images, 2).repeat(1, 3, 1, 1),
            F.avg_pool2d(images, 4).repeat(1, 4, 1, 1),
            F.adaptive_avg_pool2d(images, (1, 1)).repeat(1, 5, 1, 1),
        )
        transformed = tuple(
            one * model.scale + 0.1 * (index + 1)
            for index, one in enumerate(encoder)
        )
        evidence1 = (encoder[0], encoder[0], encoder[0])
        evidence2 = (encoder[1], encoder[1])
        return subject.ForwardLocalBranches(
            encoder=encoder,
            transformed=transformed,  # type: ignore[arg-type]
            d5=encoder[3],
            evidence1=evidence1,
            evidence2=evidence2,
        )

    return prepare


def _fake_decoder(counter: dict[str, int]):
    def decode(
        model: nn.Module,
        prepared: subject.ForwardLocalBranches,
        public_mode: str,
    ) -> torch.Tensor:
        del model
        counter["decode"] += 1
        fused = subject.fuse_public_mode(
            prepared.transformed, prepared.encoder, public_mode
        )
        output = torch.zeros(
            (fused[0].shape[0], 1, *fused[0].shape[-2:]),
            device=fused[0].device,
            dtype=fused[0].dtype,
        )
        for one in fused:
            output = output + F.interpolate(
                one.mean(dim=1, keepdim=True),
                size=output.shape[-2:],
                mode="nearest",
            )
        return torch.sigmoid(output / 8.0)

    return decode


def _run_synthetic(monkeypatch: pytest.MonkeyPatch):
    counter = {"prepare": 0, "decode": 0}
    monkeypatch.setattr(subject, "prepare_forward_local_branches", _fake_preparer(counter))
    monkeypatch.setattr(subject, "decode_forward_local_mode", _fake_decoder(counter))
    model = _TinyModel().eval()
    loader = DataLoader(_TinyDataset(), batch_size=1, shuffle=False, num_workers=0)
    state_before = subject.stage2_audit.module_state_sha256(model)
    result = subject.analyze_loaded_model(
        model,
        loader,
        torch.device("cpu"),
        ("sample_1", "sample_2"),
    )
    return model, result, counter, state_before


def test_mode_matrix_contains_only_representable_counterfactuals() -> None:
    assert len(subject.PUBLIC_MODES) == 11
    assert subject.PUBLIC_MODES[0] == "current_g0"
    assert set(subject.MODE_SPECS.values()) == {
        (0.0, ()),
        (-0.25, (0,)),
        (-0.25, (1,)),
        (-0.25, (2,)),
        (-0.25, (3,)),
        (-0.25, (0, 1, 2, 3)),
        (0.25, (0,)),
        (0.25, (1,)),
        (0.25, (2,)),
        (0.25, (3,)),
        (0.25, (0, 1, 2, 3)),
    }
    assert all("f1" not in mode and "f3" not in mode for mode in subject.PUBLIC_MODES)


def test_current_is_bitwise_production_order_and_delta_is_constant_sum() -> None:
    transformed = tuple(
        torch.tensor([[[[1.0, 1e10], [-3.0, 0.25]]]], dtype=torch.float32)
        for _ in range(4)
    )
    encoder = tuple(
        torch.tensor([[[[0.5, -1e10], [2.0, -0.125]]]], dtype=torch.float32)
        for _ in range(4)
    )
    expected = tuple((one_t + one_e) + one_e for one_t, one_e in zip(transformed, encoder))
    current = subject.fuse_public_mode(transformed, encoder, "current_g0")
    assert all(torch.equal(left, right) for left, right in zip(current, expected))

    positive = subject.fuse_public_mode(transformed, encoder, "gpos025_l2_only")
    assert torch.equal(positive[0], expected[0])
    assert torch.equal(positive[2], expected[2])
    assert torch.equal(positive[3], expected[3])
    correction = transformed[1].new_tensor(0.25) * transformed[1] - (
        transformed[1].new_tensor(0.25) * encoder[1]
    )
    assert torch.equal(positive[1], expected[1] + correction)
    binding = subject.normalize_public_mode("gpos025_l2_only")
    assert binding["transformed_coefficient_selected"] == 1.25
    assert binding["encoder_coefficient_selected"] == 1.75
    assert binding["coefficient_sum"] == 3.0


def test_padding_aware_branch_statistics_exclude_pure_padding() -> None:
    accumulator = subject.BranchStatisticsAccumulator()
    encoder = tuple(
        torch.ones((1, index + 1, 4, 4), dtype=torch.float32)
        for index in range(4)
    )
    transformed = tuple(one * 2.0 for one in encoder)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[..., 0, 0] = 1.0
    accumulator.append(transformed, encoder, target, (3, 2))
    summary = accumulator.summary()
    assert summary["valid_projection"] == "adaptive_max_pool2d_any_original_support"
    assert summary["padding_policy"] == "exclude_bins_with_no_original_pixel_support"
    for row in summary["levels"]:
        assert row["valid_spatial_location_count"] == 6
        assert row["target_spatial_location_count"] == 1
        assert row["background_spatial_location_count"] == 5
        assert row["transformed_rms"] == pytest.approx(2.0)
        assert row["encoder_rms"] == pytest.approx(1.0)
        assert row["transformed_encoder_cosine"] == pytest.approx(1.0)
        assert row["current_transformed_amplitude_share_proxy"] == pytest.approx(0.5)


def test_synthetic_audit_prepares_once_and_decodes_eleven_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, result, counter, state_before = _run_synthetic(monkeypatch)
    assert tuple(result["modes"]) == subject.PUBLIC_MODES
    assert counter == {"prepare": 2, "decode": 22}
    assert result["execution_audit"]["encoder_tpd_qfg_prepare_count"] == 2
    assert result["execution_audit"]["decoder_execution_count"] == 22
    assert result["execution_audit"]["encoder_tpd_qfg_recomputed_per_mode"] is False
    assert result["restoration_audit"]["model_state_unchanged"] is True
    assert subject.stage2_audit.module_state_sha256(model) == state_before
    assert result["probability_arrays_persisted"] is False
    assert result["feature_tensors_persisted"] is False

    current = result["modes"][subject.CURRENT_MODE]
    assert current["probability_difference_to_current"]["max_abs"] == 0.0
    assert current["probability_difference_to_current"]["absolute_difference_sum"] == 0.0
    for mode in result["modes"].values():
        fixed = mode["fixed_threshold_0_5"]
        assert fixed["threshold"] == 0.5
        for metric in (
            "pd",
            "tiny_pd",
            "fa",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "miou",
            "niou",
            "test_loss",
            "unmatched_predicted_pixels",
            "false_positive_pixels",
        ):
            assert metric in fixed
        points = mode["descriptive_pd_fa"]["points"]
        assert [point["threshold"] for point in points] == [0.5, 1.0]
        assert points[1]["selected_point_is_empty"] is True
        assert points[1]["pd"] == 0.0
        assert points[1]["fa"] == 0.0


def test_output_validator_accepts_json_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result, _, _ = _run_synthetic(monkeypatch)
    payload = {
        "schema": subject.SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "method": subject.REFERENCE_METHOD,
        "training_model_method": subject.TRAINING_MODEL_METHOD,
        "checkpoint_role": "best_pd",
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": subject.EVALUATION_PROTOCOL,
        "sweep_thresholds": [0.5, 1.0],
        "mode_order": list(subject.PUBLIC_MODES),
        **result,
        "reference_replay_audit": {
            "passed": True,
            "comparison": "current_g0_fixed_threshold_0_5_vs_existing_best_pd",
        },
        "checkpoint_binding": {
            "checkpoint": {"role": "best_pd", "sha256": "a" * 64},
            "protocol": {"payload_sha256": "b" * 64},
        },
        "reference_reuse": {"checkpoint_role": "best_pd", "sha256": "c" * 64},
        "data": {
            "split": "img_idx/test",
            "protocol_manifest": {"sha256": "d" * 64},
            "inference_order_newline_sha256": "e" * 64,
        },
        "source_sha256": {
            "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": "f" * 64
        },
        "intervention_contract": {
            "family": "GCSF_constant_sum_representable_counterfactual",
            "current_formula_operation_order": "(T+E)+E",
            "selected_correction_operation_order": "baseline+(g*T-g*E)",
            "unrepresentable_f1_t_plus_e_used_for_trigger": False,
            "unrepresentable_f3_2t_plus_e_used_for_trigger": False,
            "model_state_modified": False,
            "derived_checkpoint_written": False,
        },
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
        "feature_cache_written": False,
    }
    subject.validate_output_payload(payload)
    subject.validate_output_payload(json.loads(json.dumps(payload, allow_nan=False)))


def test_atomic_json_publication_is_write_once(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation.json"
    subject.atomic_create_json(destination, {"status": "complete"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "complete"}
    with pytest.raises(FileExistsError):
        subject.atomic_create_json(destination, {"status": "replacement"})


def test_cli_defaults_bind_both_checkpoint_roles() -> None:
    args = subject.parse_args(
        ["--dataset", "NUDT-SIRST", "--checkpoint-role", "best_pd", "--device", "cpu"]
    )
    assert args.reference_evaluation.name == "best_pd.json"
    assert "v4_tss_off_best_pd_seed42" in str(args.output)
