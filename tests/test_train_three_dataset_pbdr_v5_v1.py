from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from experiments.pbdr_v5_run_contract import epoch_selection_key
from experiments import train_three_dataset_pbdr_v5_v1 as trainer


def _metrics(intersection: int = 8) -> dict[str, object]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": 10,
        "matched_target_count": 5,
        "target_count": 5,
        "unmatched_component_pixels": 0,
        "valid_pixel_count": 100,
        "matched_tiny_target_count": 2,
        "tiny_target_count": 2,
        "niou": 0.75,
        "test_loss": 0.1,
    }


def test_frozen_scope_is_exactly_three_dataset_role_pairs() -> None:
    assert trainer.FOCUS_RUNS == {
        ("NUDT-SIRST", "best_pd"),
        ("NUAA-SIRST", "best_miou"),
        ("IRSTD-1K", "best_miou"),
    }


def test_formal_recipe_has_no_performance_margin() -> None:
    recipe = trainer._training_recipe(epochs=30, eval_every=5, batch_size=16)
    assert recipe["performance_acceptance_margin"] is None
    assert recipe["fixed_probability_threshold"] == 0.5
    assert recipe["fixed_probability_comparison"] == ">"
    assert [group["name"] for group in recipe["parameter_groups"]] == [
        "pbdr_v4",
        "outc",
        "up_decoder1",
    ]


def test_selected_state_binds_epoch_zero_and_tensor_sha() -> None:
    model = torch.nn.Conv2d(1, 1, 1)
    selected = trainer._selected_state(
        model=model,
        epoch=0,
        metrics=_metrics(),
        diagnostics={"internal": True},
        role="best_miou",
    )
    assert selected["epoch"] == 0
    assert tuple(selected["selection_key_raw"]) == epoch_selection_key(
        "best_miou", _metrics(), 0
    )
    assert len(selected["state_sha256"]) == 64


def test_candidate_manifest_never_serializes_tensor_bytes() -> None:
    candidate = {
        "schema": trainer.CANDIDATE_SCHEMA,
        "dataset": "NUAA-SIRST",
        "role": "best_miou",
        "arm": trainer.ARM,
        "epoch": 0,
        "validation_metrics": _metrics(),
        "selection_key": [],
        "run_identity": {},
        "run_identity_sha256": "a" * 64,
        "run_protocol_sha256": "b" * 64,
        "v5_source_sha256": "c" * 64,
        "loss_manifest_sha256": "d" * 64,
        "stage1_checkpoint_sha256": "e" * 64,
        "stage1_state_sha256": "f" * 64,
        "v4_compatible_candidate": {
            "state_sha256": "0" * 64,
            "state_dict": {"weight": torch.ones(1)},
        },
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
    }
    manifest = trainer._candidate_manifest(candidate)
    assert "state_dict" not in repr(manifest)
    assert manifest["state_sha256"] == "0" * 64


def test_run_rejects_out_of_scope_pair_before_any_io() -> None:
    args = Namespace(dataset="NUDT-SIRST", role="best_miou")
    with pytest.raises(trainer.PBDRV5TrainerError, match="outside"):
        trainer.run(args)


def test_parser_requires_internal_bindings() -> None:
    with pytest.raises(SystemExit):
        trainer.parse_args(["--dataset", "NUAA-SIRST", "--role", "best_miou"])
