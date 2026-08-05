from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from analysis import analyze_ner_stage2_mask_knockout_v1 as analyzer
from analysis import compare_ner_stage2_mask_knockout_v1 as comparator


class _FakeComplementRelay(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.dc_support_mode = "complement_tail"
        self.gates = nn.ModuleDict({"2": nn.Identity()})

    def dc_support(self, stage, relay_value, sources, output_size):
        del stage, sources, output_size
        return torch.full_like(relay_value[:, :1], 0.75)

    def forward_stage(self, stage, sources, output_size):
        del output_size
        value = sources[0] * self.weight
        mask = torch.full_like(value[:, :1], float(stage) / 10.0)
        return value, mask


def test_methodtype_knockout_changes_only_stage2_return_and_restores_method():
    relay = _FakeComplementRelay()
    recorder = analyzer.Stage2KnockoutRecorder()
    state_before = analyzer.module_state_sha256(relay)
    source = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    class_forward = relay.forward_stage.__func__

    with analyzer.temporary_stage2_mask_knockout(relay, recorder):
        value4, mask4 = relay.forward_stage(4, (source,), (2, 2))
        value3, mask3 = relay.forward_stage(3, (source,), (2, 2))
        value2, mask2 = relay.forward_stage(2, (source,), (2, 2))

    assert relay.forward_stage.__func__ is class_forward
    assert analyzer.module_state_sha256(relay) == state_before
    assert torch.equal(value4, source * 2.0)
    assert torch.equal(value3, source * 2.0)
    assert torch.equal(value2, source * 2.0)
    assert torch.equal(mask4, torch.full_like(mask4, 0.4))
    assert torch.equal(mask3, torch.full_like(mask3, 0.3))
    assert torch.count_nonzero(mask2).item() == 0
    assert recorder.stage_call_counts == {4: 1, 3: 1, 2: 1}
    assert recorder.returned_stage2_mask_abs_max == 0.0
    assert len(recorder.observations) == 1
    np.testing.assert_array_equal(
        recorder.observations[0].original_mask,
        np.full((2, 2), 0.2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        recorder.observations[0].persistent_support_p2,
        np.full((2, 2), 0.25, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        recorder.observations[0].centered_local_logits,
        np.array([[-3.0, -1.0], [1.0, 3.0]], dtype=np.float32),
    )


def test_methodtype_knockout_restores_method_after_exception():
    relay = _FakeComplementRelay()
    recorder = analyzer.Stage2KnockoutRecorder()
    class_forward = relay.forward_stage.__func__
    with pytest.raises(RuntimeError, match="synthetic"):
        with analyzer.temporary_stage2_mask_knockout(relay, recorder):
            raise RuntimeError("synthetic")
    assert relay.forward_stage.__func__ is class_forward
    assert "forward_stage" not in relay.__dict__


def test_unmatched_component_mask_mirrors_centroid_matching():
    target = np.zeros((8, 8), dtype=np.float32)
    target[1, 1] = 1.0
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[1, 1] = 0.9
    probability[7, 7] = 0.9

    false_mask, counts = analyzer.unmatched_false_component_mask(
        probability, target
    )

    assert false_mask.sum() == 1
    assert false_mask[7, 7]
    assert not false_mask[1, 1]
    assert counts == {
        "predicted_object_count": 2,
        "target_count": 1,
        "matched_target_count": 1,
        "unmatched_predicted_object_count": 1,
        "unmatched_predicted_pixels": 1,
    }


def test_pixel_confusion_and_reference_recovery_are_integer_exact():
    target = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    probability = np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    confusion = analyzer.pixel_confusion([probability], [target])
    assert confusion["true_positive_pixels"] == 1
    assert confusion["false_positive_pixels"] == 1
    assert confusion["false_negative_pixels"] == 1
    assert confusion["true_negative_pixels"] == 1

    recovered = analyzer.recover_reference_pixel_confusion(
        {"pixel_precision": 0.5, "pixel_recall": 0.5},
        target_positive_pixels=2,
        valid_pixel_count=4,
    )
    assert recovered["recoverable"] is True
    assert recovered["true_positive_pixels"] == 1
    assert recovered["false_positive_pixels"] == 1
    assert recovered["false_negative_pixels"] == 1
    assert recovered["true_negative_pixels"] == 1


def test_mechanism_statistics_freeze_cell_mapping_and_gate_raw_values():
    target = np.zeros((4, 4), dtype=np.float32)
    target[0, 0] = 1.0
    probability = np.zeros((4, 4), dtype=np.float32)
    probability[0, 0] = 0.9
    probability[3, 3] = 0.9
    observation = analyzer.Stage2Observation(
        original_mask=np.array([[0.1, 0.1], [0.1, 0.4]], dtype=np.float32),
        persistent_support_p2=np.array(
            [[0.8, 0.9], [0.9, 0.2]], dtype=np.float32
        ),
        centered_local_logits=np.array(
            [[-0.6, 0.1], [0.1, 0.4]], dtype=np.float32
        ),
        output_size=(2, 2),
    )

    result = analyzer.analyze_stage2_observations(
        [probability], [target], [observation]
    )

    b = result["gate_b_raw"]
    assert b["available"] is False
    assert b["reason"] == "reference_probability_cache_absent"
    descriptive = b["knockout_region_descriptive_only"]
    assert descriptive["false_component_density"] == pytest.approx(0.4)
    assert descriptive["normal_background_density"] == pytest.approx(0.1)
    assert descriptive["density_ratio"] == pytest.approx(4.0)
    c = result["gate_c_raw"]
    assert c["available"] is True
    assert c["low_p2_threshold"] == 0.25
    assert c["mass_share"] == pytest.approx(2.0 / 3.0)
    assert result["regions"]["gt_target_cells"]["cell_count"] == 1
    assert (
        result["regions"]["knockout_unmatched_false_component_cells"][
            "cell_count"
        ]
        == 1
    )
    assert result["positive_local_signal_definition"] == "relu(centered_local_logits)"
    assert result["original_final_mask_descriptive"]["excluded_from_gate_b_and_c"]
    assert result["full_probability_arrays_persisted"] is False


def test_gate_a_raw_uses_relative_reductions_and_exact_drops():
    reference = {
        "unmatched_predicted_pixels": 100,
        "matched_target_count": 20,
        "matched_tiny_target_count": 5,
        "miou": 0.8,
        "niou": 0.7,
    }
    knockout = {
        "unmatched_predicted_pixels": 90,
        "matched_target_count": 19,
        "matched_tiny_target_count": 5,
        "miou": 0.797,
        "niou": 0.698,
    }
    reference_pixel = {"recoverable": True, "false_positive_pixels": 200}
    knockout_pixel = {"false_positive_pixels": 180}
    result = analyzer.build_gate_a_raw(
        reference, knockout, reference_pixel, knockout_pixel
    )
    assert result["component_fa_relative_reduction"] == pytest.approx(0.1)
    assert result["all_background_pixel_fp_relative_reduction"] == pytest.approx(0.1)
    assert result["matched_target_drop"] == 1
    assert result["matched_tiny_target_drop"] == 0
    assert result["miou_drop"] == pytest.approx(0.003)
    assert result["niou_drop"] == pytest.approx(0.002)


def test_historical_loader_compatibility_restores_core_validator():
    original = analyzer.core._validate_training_runtime_sources
    with analyzer.historical_checkpoint_loader_compatibility():
        assert (
            analyzer.core._validate_training_runtime_sources
            is analyzer._validate_historical_runtime_sources_allow_additions
        )
    assert analyzer.core._validate_training_runtime_sources is original


def test_historical_source_validator_accepts_real_protocol_shape_and_new_files():
    protocol_path = (
        analyzer.REPO_ROOT
        / "results"
        / "three_dataset_tss_off_seed42_v1"
        / "runs"
        / "NUAA-SIRST"
        / "final_tss_off"
        / "seed_42"
        / "protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen = protocol["runtime_sources"]
    expected_non_architecture = {
        "data_protocol",
        "model_builder",
        "protocol_document",
        "reused_positive_runner",
        "runner",
        "torch_datasets",
        "training_engine",
        "training_loss",
        "training_metrics_and_schedule",
    }
    assert {
        key for key in frozen if not key.startswith("architecture::")
    } == expected_non_architecture

    verified = analyzer._validate_historical_runtime_sources_allow_additions(
        protocol
    )
    assert verified == {
        key: entry["sha256"] for key, entry in sorted(frozen.items())
    }
    assert len(verified) == 35

    new_v5 = analyzer.REPO_ROOT / "model" / "tpd_ner_v8_mprs_dch_v5_per.py"
    assert new_v5.is_file()
    assert "architecture::model/tpd_ner_v8_mprs_dch_v5_per.py" not in frozen
    # The current unlisted V5 source neither enters nor invalidates the exact
    # set of historical dependencies returned above.
    assert all(Path(entry["path"]).resolve() != new_v5.resolve() for entry in frozen.values())

    tampered = json.loads(json.dumps(protocol))
    tampered["runtime_sources"]["data_protocol"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime source SHA differs: data_protocol"):
        analyzer._validate_historical_runtime_sources_allow_additions(tampered)


def test_atomic_json_refuses_implicit_overwrite(tmp_path: Path):
    output = tmp_path / "result.json"
    analyzer.atomic_write_json(output, {"value": 1}, overwrite=False)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError):
        analyzer.atomic_write_json(output, {"value": 2}, overwrite=False)


def _synthetic_comparison_payload(
    dataset: str,
    *,
    a_reduction: float = 0.06,
    target_drop: int = 1,
    tiny_drop: int = 1,
    miou_drop: float = 0.004,
    niou_drop: float = 0.004,
    b_available: bool = False,
    b_ratio: float | None = None,
    c_share: float = 0.30,
    c_denominator_zero: bool = False,
):
    return {
        "schema": analyzer.SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "checkpoint_role": "best_miou",
        "seed": 42,
        "intervention": analyzer.INTERVENTION,
        "fixed_threshold_0_5": {"threshold": 0.5},
        "intervention_audit": {
            "model_state_unchanged": True,
            "returned_stage2_mask_abs_max": 0.0,
        },
        "gate_inputs": {
            "A": {
                "component_fa_relative_reduction": a_reduction,
                "all_background_pixel_fp_relative_reduction": None,
                "matched_target_drop": target_drop,
                "matched_tiny_target_drop": tiny_drop,
                "miou_drop": miou_drop,
                "niou_drop": niou_drop,
            },
            "B": {
                "available": b_available,
                "reason": None if b_available else "reference_probability_cache_absent",
                "density_ratio": b_ratio,
                "denominator_is_zero": False,
            },
            "C": {
                "available": not c_denominator_zero,
                "low_p2_threshold": 0.25,
                "mass_share": c_share,
                "denominator_is_zero": c_denominator_zero,
            },
        },
    }


def test_comparator_authorizes_only_when_a_and_c_each_pass_two_of_three():
    payloads = {
        comparator.DATASETS[0]: _synthetic_comparison_payload(
            comparator.DATASETS[0]
        ),
        comparator.DATASETS[1]: _synthetic_comparison_payload(
            comparator.DATASETS[1]
        ),
        comparator.DATASETS[2]: _synthetic_comparison_payload(
            comparator.DATASETS[2], a_reduction=0.01, c_share=0.10
        ),
    }
    result = comparator.compare_payloads(payloads)
    assert result["aggregate_gates"]["A"]["passed_dataset_count"] == 2
    assert result["aggregate_gates"]["B"]["passed_dataset_count"] == 0
    assert result["aggregate_gates"]["C"]["passed_dataset_count"] == 2
    assert result["ner_v5_per_development_training_authorized"] is True
    assert result["decision"] == "AUTHORIZE_ONE_NER_V5_PER_DEVELOPMENT_CANDIDATE"


def test_comparator_thresholds_use_inclusive_improvement_and_exclusive_drops():
    dataset = comparator.DATASETS[0]
    boundary_pass = _synthetic_comparison_payload(
        dataset,
        a_reduction=0.05,
        target_drop=1,
        tiny_drop=1,
        miou_drop=0.004999,
        niou_drop=0.004999,
        c_share=0.25,
    )
    evaluated = comparator.evaluate_dataset_gate(boundary_pass)
    assert evaluated["A"]["pass"] is True
    assert evaluated["C"]["pass"] is True

    target_boundary_fail = _synthetic_comparison_payload(dataset, target_drop=2)
    assert comparator.evaluate_dataset_gate(target_boundary_fail)["A"]["pass"] is False
    miou_boundary_fail = _synthetic_comparison_payload(dataset, miou_drop=0.005)
    assert comparator.evaluate_dataset_gate(miou_boundary_fail)["A"]["pass"] is False


def test_comparator_zero_denominator_never_passes_c():
    payload = _synthetic_comparison_payload(
        comparator.DATASETS[0], c_share=1.0, c_denominator_zero=True
    )
    evaluated = comparator.evaluate_dataset_gate(payload)
    assert evaluated["C"]["available"] is False
    assert evaluated["C"]["pass"] is False


def test_comparator_can_apply_b_only_when_reference_aligned_input_is_available():
    payloads = {
        dataset: _synthetic_comparison_payload(
            dataset,
            b_available=index < 2,
            b_ratio=1.25 if index < 2 else None,
            c_share=0.10,
        )
        for index, dataset in enumerate(comparator.DATASETS)
    }
    result = comparator.compare_payloads(payloads)
    assert result["aggregate_gates"]["A"]["pass"] is True
    assert result["aggregate_gates"]["B"]["pass"] is True
    assert result["aggregate_gates"]["C"]["pass"] is False
    assert result["ner_v5_per_development_training_authorized"] is True


def test_comparator_rejects_incomplete_dataset_matrix():
    one = comparator.DATASETS[0]
    with pytest.raises(ValueError, match="exactly three"):
        comparator.compare_payloads({one: _synthetic_comparison_payload(one)})
