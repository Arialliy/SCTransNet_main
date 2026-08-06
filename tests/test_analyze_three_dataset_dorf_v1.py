from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from analysis import analyze_three_dataset_dorf_v1 as analyzer


class TinyDORFDataset(Dataset):
    def __init__(self) -> None:
        self.sample_ids = ["sample_a", "sample_b"]
        self.images = [
            torch.tensor(
                [[[-2.0, -1.0, -1.0, -2.0],
                  [-1.0, 2.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0, -1.0],
                  [-2.0, -1.0, -1.0, -2.0]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[-1.0, -1.0, -1.0, -1.0],
                  [-1.0, -1.0, -1.0, -1.0],
                  [-1.0, -1.0, 2.0, -1.0],
                  [-1.0, -1.0, -1.0, -1.0]]],
                dtype=torch.float32,
            ),
        ]
        self.masks = [
            torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[[0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        ]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return (
            self.images[index],
            self.masks[index],
            (4, 4),
            self.sample_ids[index],
        )


class TinyDORFModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outc = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.outconv = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.outc.weight.fill_(1.0)
            self.outc.bias.zero_()
            self.outconv.weight.fill_(0.4)
            self.outconv.bias.fill_(-0.25)
        self.forward_count = 0
        self.mode = "test"
        self.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        z_out = self.outc(image)
        z_d0 = self.outconv(torch.cat([z_out] * 5, dim=1))
        # Keep a live dependency on z_d0 while preserving the formal output.
        return torch.sigmoid(z_out + z_d0 * 0.0)


class DuplicateOutModel(TinyDORFModel):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        first = self.outc(image)
        second = self.outc(image)
        self.outconv(torch.cat([first] * 5, dim=1))
        return torch.sigmoid(second)


class MissingD0Model(TinyDORFModel):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        return torch.sigmoid(self.outc(image))


class WrongReturnedProbabilityModel(TinyDORFModel):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        z_out = self.outc(image)
        self.outconv(torch.cat([z_out] * 5, dim=1))
        return torch.sigmoid(z_out + 0.1)


def _analyzed() -> tuple[TinyDORFModel, dict]:
    dataset = TinyDORFDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = TinyDORFModel()
    output = analyzer.analyze_loaded_model(
        model, loader, torch.device("cpu"), dataset.sample_ids
    )
    return model, output


def _valid_payload(analyzed: dict) -> dict:
    manifest = json.loads(
        analyzer.DEFAULT_INPUT_MANIFEST.read_text(encoding="utf-8")
    )
    entry = next(
        row
        for row in manifest["entries"]
        if row["method"] == "final_tss_off"
        and row["dataset"] == "NUDT-SIRST"
        and row["checkpoint_role"] == "best_miou"
    )
    reference_sha = entry["evaluation_sha256"]
    state_sha = "b" * 64
    engineering = copy.deepcopy(analyzed["engineering_audit"])
    engineering.update(
        {
            "source_sha256_reverified_after_inference": True,
            "input_manifest_reverified_after_inference": True,
            "alpha0_historical_replay_passed": True,
        }
    )
    background_sha = manifest["background_pixel_authority"]["sha256"]
    run_dir = (analyzer.REPO_ROOT / entry["run_dir"]).resolve()
    checkpoint_path = run_dir / "checkpoints" / "best_miou.pth.tar"
    evaluation_path = run_dir / "evaluations" / "best_miou.json"
    data_manifest_path = str(
        (
            analyzer.REPO_ROOT
            / manifest["data_protocol_manifest"]["path"]
        ).resolve()
    )
    return {
        "schema": analyzer.SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "method": "final_tss_off",
        "checkpoint_role": "best_miou",
        "seed": analyzer.SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        "mode_order": list(analyzer.MODE_ORDER),
        "modes": analyzed["modes"],
        "input_manifest_binding": {
            "path": str(analyzer.DEFAULT_INPUT_MANIFEST.resolve()),
            "sha256": analyzer.FROZEN_INPUT_MANIFEST_SHA256,
            "schema": analyzer.INPUT_MANIFEST_SCHEMA,
            "status": "frozen_before_dorf_outputs",
            "entry_key": "final_tss_off::NUDT-SIRST::best_miou",
            "entry": entry,
            "data_protocol_manifest": {
                "path": data_manifest_path,
                "sha256": manifest["data_protocol_manifest"]["sha256"],
            },
            "background_pixel_authority": {
                "path": str(
                    (
                        analyzer.REPO_ROOT
                        / manifest["background_pixel_authority"]["path"]
                    ).resolve()
                ),
                "sha256": background_sha,
            },
            "historical_metric_authority": "bound_evaluation_json_only",
            "checkpoint_embedded_metrics_fallback_allowed": False,
            "verified_before_model_load": True,
            "verified_after_inference": True,
        },
        "checkpoint_binding": {
            "checkpoint": {
                "role": "best_miou",
                "sha256": entry["checkpoint_sha256"],
                "epoch": entry["epoch"],
                "path": str(checkpoint_path),
            },
            "training_state_dict_sha256": state_sha,
            "input_manifest_entry_key": "final_tss_off::NUDT-SIRST::best_miou",
            "run_dir": str(run_dir),
            "summary": {
                "path": str(run_dir / "summary.json"),
                "sha256": entry["summary_sha256"],
            },
            "protocol": {
                "path": str(run_dir / "protocol.json"),
                "sha256": entry["protocol_sha256"],
            },
        },
        "reference_evaluation_binding": {
            "checkpoint_role": "best_miou",
            "path": str(evaluation_path),
            "sha256": reference_sha,
            "source": "historical_evaluation_fixed_threshold_0_5",
            "checkpoint_embedded_metrics_fallback_allowed": False,
        },
        "source_sha256": {"analysis/test.py": "d" * 64},
        "model_metadata": {
            "strict_load": True,
            "dorf_loader_audit": {
                "passed": True,
                "builder": analyzer.EXPECTED_BUILDERS["final_tss_off"],
                "training_state_key_count": 568,
                "expected_training_state_key_count": 568,
                "removed_training_only_tss_state_key_count": 4,
                "inference_state_key_count": 564,
                "strict_load": True,
                "training_flag": False,
                "mode": "test",
            },
        },
        "background_pixel_authority_record": {
            "dataset": "NUDT-SIRST",
            "checkpoint_role": "best_miou",
            "checkpoint_epoch": entry["epoch"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "evaluation_sha256": entry["evaluation_sha256"],
            "false_positive_pixels": analyzed["modes"]["current_out"][
                "fixed_threshold_0_5"
            ]["false_positive_pixels"],
            "valid_pixel_count": analyzed["modes"]["current_out"][
                "fixed_threshold_0_5"
            ]["valid_pixel_count"],
        },
        "alpha0_historical_replay_audit": {
            "passed": True,
            "exact": True,
            "counts_exact": True,
            "background_false_positive_pixels_exact": True,
            "within_frozen_float_tolerances": True,
            "mode": "current_out",
            "checkpoint_role": "best_miou",
            "reference_evaluation_sha256": reference_sha,
            "background_pixel_authority_sha256": background_sha,
        },
        "engineering_audit": engineering,
        "data": {
            "split": "img_idx/test",
            "protocol_manifest": {
                "path": data_manifest_path,
                "sha256": manifest["data_protocol_manifest"]["sha256"],
            },
            "input_binding": analyzed["input_binding"],
        },
        "intervention_contract": {
            "family": "DORF_V1_existing_deep_supervision_readout_reuse",
            "formula": "z_out + alpha * (z_d0 - z_out)",
            "fusion_space": "raw_logits_before_sigmoid",
            "alphas": [analyzer.MODE_ALPHA[mode] for mode in analyzer.MODE_ORDER],
            "one_checkpoint_per_unit": True,
            "model_parameters_changed": False,
            "persistent_buffers_changed": False,
            "derived_checkpoint_written": False,
        },
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
    }


def test_frozen_modes_and_default_paths() -> None:
    assert analyzer.MODE_ALPHA == {
        "current_out": 0.0,
        "dorf_a025": 0.25,
        "dorf_a050": 0.5,
        "dorf_a075": 0.75,
        "d0_only": 1.0,
    }
    final = analyzer.parse_args(
        [
            "--method",
            "final_tss_off",
            "--dataset",
            "NUDT-SIRST",
            "--checkpoint-role",
            "best_pd",
            "--device",
            "cpu",
        ]
    )
    assert final.input_manifest == analyzer.DEFAULT_INPUT_MANIFEST
    assert final.run_dir is None
    assert final.reference_evaluation is None
    assert final.output == (
        analyzer.DEFAULT_OUTPUT_ROOT
        / "runs/final_tss_off/NUDT-SIRST/best_pd/evaluation.json"
    )
    original = analyzer.parse_args(
        [
            "--method",
            "original",
            "--dataset",
            "NUAA-SIRST",
            "--checkpoint-role",
            "best_miou",
        ]
    )
    assert original.run_dir is None
    assert original.output == (
        analyzer.DEFAULT_OUTPUT_ROOT
        / "runs/original/NUAA-SIRST/best_miou/evaluation.json"
    )


def test_requests_bind_each_historical_training_schema() -> None:
    final = analyzer._request_for("final_tss_off", "NUDT-SIRST", "best_miou")
    assert final.method == "final"
    assert final.requested_tss_weight == 0.0
    assert analyzer.core.TRAINING_RUN_SCHEMA == analyzer.tss_off_adapter.TRAINING_RUN_SCHEMA
    original = analyzer._request_for("original", "NUDT-SIRST", "best_pd")
    assert original.method == "original"
    assert original.requested_tss_weight is None
    assert analyzer.core.TRAINING_RUN_SCHEMA == analyzer.ORIGINAL_TRAINING_RUN_SCHEMA


def test_fuse_raw_logits_uses_preregistered_logit_formula() -> None:
    z_out = torch.tensor([[[[-2.0, 0.5]]]], dtype=torch.float32)
    z_d0 = torch.tensor([[[[2.0, -1.5]]]], dtype=torch.float32)
    assert analyzer.fuse_raw_logits(z_out, z_d0, 0.0) is z_out
    assert analyzer.fuse_raw_logits(z_out, z_d0, 1.0) is z_d0
    assert torch.equal(
        analyzer.fuse_raw_logits(z_out, z_d0, 0.25),
        z_out + 0.25 * (z_d0 - z_out),
    )
    with pytest.raises(ValueError, match="not preregistered"):
        analyzer.fuse_raw_logits(z_out, z_d0, 0.1)
    with pytest.raises(ValueError, match="shapes differ"):
        analyzer.fuse_raw_logits(z_out, z_d0[:, :, :, :1], 0.5)


def test_single_forward_captures_each_logit_once_and_builds_all_modes() -> None:
    model, analyzed = _analyzed()
    assert model.forward_count == 2
    assert not model.outc._forward_hooks
    assert not model.outconv._forward_hooks
    assert list(analyzed["modes"]) == list(analyzer.MODE_ORDER)
    engineering = analyzed["engineering_audit"]
    assert engineering["passed"] is True
    assert engineering["batch_count"] == 2
    assert engineering["model_forward_count"] == 2
    assert engineering["outc_hook_count"] == 2
    assert engineering["outconv_hook_count"] == 2
    assert engineering["same_d0_out_logits_reused_for_all_modes"] is True
    assert engineering["model_state_sha256_before"] == engineering[
        "model_state_sha256_after"
    ]
    assert len(analyzed["input_binding"]["sha256"]) == 64

    current = analyzed["modes"]["current_out"]
    difference = current["probability_difference_to_current"]
    assert difference["element_count"] == 32
    assert difference["absolute_difference_sum"] == 0.0
    assert difference["max_abs"] == 0.0
    assert difference["mean_abs"] == 0.0
    assert analyzed["modes"]["d0_only"][
        "probability_difference_to_current"
    ]["max_abs"] > 0.0

    for mode in analyzer.MODE_ORDER:
        record = analyzed["modes"][mode]
        assert record["mode"] == mode
        assert record["alpha"] == analyzer.MODE_ALPHA[mode]
        fixed = record["fixed_threshold_0_5"]
        assert analyzer._REQUIRED_FIXED_FIELDS <= set(fixed)
        assert fixed["image_count"] == 2
        assert fixed["valid_pixel_count"] == 32
        points = record["descriptive_pd_fa"]["points"]
        assert [point["threshold"] for point in points] == [0.5, 1.0]
        assert points[1]["selected_point_is_empty"] is True
        assert points[1]["pd"] == 0.0
        assert points[1]["fa"] == 0.0


def test_current_out_bypasses_fusion_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = TinyDORFDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = TinyDORFModel()
    observed_alphas: list[float] = []
    original = analyzer.fuse_raw_logits

    def recording_fusion(z_out, z_d0, alpha):
        observed_alphas.append(float(alpha))
        return original(z_out, z_d0, alpha)

    monkeypatch.setattr(analyzer, "fuse_raw_logits", recording_fusion)
    analyzer.analyze_loaded_model(
        model, loader, torch.device("cpu"), dataset.sample_ids
    )
    assert 0.0 not in observed_alphas
    assert observed_alphas == [0.25, 0.5, 0.75, 1.0] * len(dataset)


def test_model_must_enter_and_leave_as_eval_mode_test() -> None:
    dataset = TinyDORFDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    training_model = TinyDORFModel()
    training_model.train()
    with pytest.raises(ValueError, match="already be in eval mode"):
        analyzer.analyze_loaded_model(
            training_model, loader, torch.device("cpu"), dataset.sample_ids
        )
    wrong_mode = TinyDORFModel()
    wrong_mode.mode = "train"
    with pytest.raises(ValueError, match="mode must remain test"):
        analyzer.analyze_loaded_model(
            wrong_mode, loader, torch.device("cpu"), dataset.sample_ids
        )


@pytest.mark.parametrize(
    ("model_type", "message"),
    [
        (DuplicateOutModel, "out executed more than once"),
        (MissingD0Model, "did not each execute once"),
        (WrongReturnedProbabilityModel, "not exact sigmoid"),
    ],
)
def test_hook_and_current_output_contracts_fail_closed(model_type, message: str) -> None:
    dataset = TinyDORFDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = model_type()
    with pytest.raises(ValueError, match=message):
        analyzer.analyze_loaded_model(
            model, loader, torch.device("cpu"), dataset.sample_ids
        )
    assert not model.outc._forward_hooks
    assert not model.outconv._forward_hooks


def test_probability_difference_counts_every_unpadded_pixel() -> None:
    current = [np.zeros((2, 3), dtype=np.float32)]
    other = [np.full((2, 3), 0.25, dtype=np.float32)]
    difference = analyzer.probability_difference(current, other)
    assert difference["element_count"] == 6
    assert difference["absolute_difference_sum"] == 1.5
    assert difference["mean_abs"] == 0.25
    assert difference["max_abs"] == 0.25
    assert difference["functionally_different"] is True


def test_alpha0_replay_is_strict_and_bound_to_reference_sha() -> None:
    _, analyzed = _analyzed()
    observed = analyzed["modes"]["current_out"]["fixed_threshold_0_5"]
    reference = {
        key: value
        for key, value in observed.items()
        if key not in {"false_positive_pixels", "image_count"}
    }
    audit = analyzer.alpha0_historical_replay_audit(
        observed,
        reference,
        checkpoint_role="best_miou",
        reference_evaluation_sha256="e" * 64,
        expected_background_false_positive_pixels=observed[
            "false_positive_pixels"
        ],
        background_pixel_authority_sha256="f" * 64,
    )
    assert audit["passed"] is True
    assert audit["exact"] is True
    assert audit["reference_evaluation_sha256"] == "e" * 64
    changed = copy.deepcopy(reference)
    changed["miou"] = float(changed["miou"]) + 1e-12
    tolerant = analyzer.alpha0_historical_replay_audit(
        observed,
        changed,
        checkpoint_role="best_miou",
        reference_evaluation_sha256="e" * 64,
        expected_background_false_positive_pixels=observed[
            "false_positive_pixels"
        ],
        background_pixel_authority_sha256="f" * 64,
    )
    assert tolerant["passed"] is True
    assert tolerant["exact"] is False
    changed["miou"] = float(reference["miou"]) + 2e-4
    with pytest.raises(ValueError, match="exceeds frozen tolerance"):
        analyzer.alpha0_historical_replay_audit(
            observed,
            changed,
            checkpoint_role="best_miou",
            reference_evaluation_sha256="e" * 64,
            expected_background_false_positive_pixels=observed[
                "false_positive_pixels"
            ],
            background_pixel_authority_sha256="f" * 64,
        )


def test_output_validator_enforces_engineering_and_current_identity() -> None:
    _, analyzed = _analyzed()
    payload = _valid_payload(analyzed)
    analyzer.validate_output_payload(payload)

    bad_forward = copy.deepcopy(payload)
    bad_forward["engineering_audit"]["model_forward_count"] += 1
    with pytest.raises(ValueError, match="hook count contract"):
        analyzer.validate_output_payload(bad_forward)

    bad_current = copy.deepcopy(payload)
    bad_current["modes"]["current_out"][
        "probability_difference_to_current"
    ]["max_abs"] = 1e-8
    with pytest.raises(ValueError, match="self-difference"):
        analyzer.validate_output_payload(bad_current)

    bad_source = copy.deepcopy(payload)
    bad_source["source_sha256"] = {"analysis/test.py": "bad"}
    with pytest.raises(ValueError, match="source SHA"):
        analyzer.validate_output_payload(bad_source)


def test_frozen_manifest_binds_selected_inputs_and_background_authority() -> None:
    binding, resolved, background = analyzer.load_input_manifest_binding(
        input_manifest=analyzer.DEFAULT_INPUT_MANIFEST,
        method="final_tss_off",
        dataset="NUDT-SIRST",
        checkpoint_role="best_miou",
        run_dir=None,
        reference_evaluation=None,
        data_protocol_manifest=analyzer.data_protocol.DEFAULT_MANIFEST_PATH,
    )
    assert binding["sha256"] == analyzer.FROZEN_INPUT_MANIFEST_SHA256
    assert binding["entry_key"] == "final_tss_off::NUDT-SIRST::best_miou"
    assert binding["historical_metric_authority"] == "bound_evaluation_json_only"
    assert binding["checkpoint_embedded_metrics_fallback_allowed"] is False
    assert binding["verified_before_model_load"] is True
    assert binding["verified_after_inference"] is False
    assert analyzer.file_sha256(resolved["evaluation"]) == binding["entry"][
        "evaluation_sha256"
    ]
    assert background["recipe"] == "tss_off"
    assert background["false_positive_pixels"] >= 0


def test_manifest_path_is_frozen_not_dynamically_discovered(tmp_path: Path) -> None:
    copied = tmp_path / "dorf_v1_input_manifest.json"
    copied.write_bytes(analyzer.DEFAULT_INPUT_MANIFEST.read_bytes())
    with pytest.raises(ValueError, match="manifest path differs"):
        analyzer.load_input_manifest_binding(
            input_manifest=copied,
            method="original",
            dataset="NUDT-SIRST",
            checkpoint_role="best_pd",
            run_dir=None,
            reference_evaluation=None,
            data_protocol_manifest=analyzer.data_protocol.DEFAULT_MANIFEST_PATH,
        )


def test_atomic_json_is_write_once_and_never_contains_arrays(tmp_path: Path) -> None:
    _, analyzed = _analyzed()
    payload = _valid_payload(analyzed)
    output = tmp_path / "evaluation.json"
    analyzer.atomic_create_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == analyzer.SCHEMA
    serialized = output.read_text(encoding="utf-8")
    assert "probabilities" not in serialized
    with pytest.raises(FileExistsError, match="refusing existing output"):
        analyzer.atomic_create_json(output, payload)


def test_analyzer_uses_added_source_compatible_historical_loader() -> None:
    source = Path(analyzer.__file__).read_text(encoding="utf-8")
    assert "checkpoint_compat._load_checkpoint_allowing_added_sources" in source
    assert "core.load_checkpoint(" not in source
