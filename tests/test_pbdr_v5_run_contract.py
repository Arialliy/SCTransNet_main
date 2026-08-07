from __future__ import annotations

import copy

import pytest
import torch

from experiments.pbdr_v4_training_core import capture_rng_state
from experiments.pbdr_v5_run_contract import (
    ARM,
    PBDRV5RunContractError,
    V5RunIdentity,
    build_rolling_payload,
    epoch_selection_key,
    json_selection_key,
    validate_rolling_payload,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64
_E = "e" * 64
_F = "f" * 64
_ZERO = "0" * 64


def _identity(**updates: str) -> V5RunIdentity:
    values = {
        "dataset": "NUAA-SIRST",
        "role": "best_miou",
        "arm": ARM,
        "v4_source_lock_sha256": _A,
        "split_projection_sha256": _B,
        "atlas_manifest_sha256": _C,
        "parent_checkpoint_sha256": _D,
        "parent_state_sha256": _E,
        "stage1_checkpoint_sha256": _F,
        "stage1_state_sha256": _ZERO,
        "v5_source_sha256": "1" * 64,
        "loss_manifest_sha256": "2" * 64,
        "trainable_parameter_names_sha256": "3" * 64,
    }
    values.update(updates)
    return V5RunIdentity(**values)


def _metrics() -> dict[str, object]:
    return {
        "intersection_pixels": 8,
        "union_pixels": 10,
        "matched_target_count": 4,
        "target_count": 5,
        "unmatched_component_pixels": 2,
        "valid_pixel_count": 100,
        "matched_tiny_target_count": 1,
        "tiny_target_count": 2,
        "niou": 0.75,
        "test_loss": 0.1,
    }


def _payload() -> tuple[dict[str, object], V5RunIdentity, list[dict[str, object]]]:
    identity = _identity()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    metrics = _metrics()
    key = epoch_selection_key(identity.role, metrics, 0)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    from experiments.pbdr_v4_state_contract import state_semantic_sha256
    from experiments.pbdr_v4_run_artifacts import optimizer_group_signature

    selected = {
        "epoch": 0,
        "metrics": metrics,
        "diagnostics": {},
        "selection_key": json_selection_key(key),
        "selection_key_raw": key,
        "state_dict": state,
        "state_sha256": state_semantic_sha256(state),
    }
    payload = build_rolling_payload(
        identity=identity,
        epoch=1,
        epochs=30,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        rng_state=capture_rng_state(),
        selected=selected,
        evaluation_history=[{"epoch": 0, "metrics": metrics}],
        event={"epoch": 1},
    )
    signature = optimizer_group_signature(optimizer.state_dict())
    return payload, identity, signature


def test_epoch_zero_wins_an_exact_metric_tie() -> None:
    metrics = _metrics()
    assert epoch_selection_key("best_miou", metrics, 0) > epoch_selection_key(
        "best_miou", metrics, 1
    )


def test_epoch_selection_has_no_positive_margin() -> None:
    low = _metrics()
    high = dict(low)
    high["intersection_pixels"] = 9
    assert epoch_selection_key("best_miou", high, 30) > epoch_selection_key(
        "best_miou", low, 0
    )


def test_valid_rolling_payload_replays() -> None:
    payload, identity, signature = _payload()
    validated = validate_rolling_payload(
        payload,
        identity=identity,
        epochs=30,
        expected_optimizer_group_signature=signature,
    )
    assert validated["selected"]["epoch"] == 0
    assert validated["official_test_accessed"] is False
    assert validated["performance_acceptance_margin"] is None


@pytest.mark.parametrize(
    "field",
    (
        "v5_source_sha256",
        "loss_manifest_sha256",
        "trainable_parameter_names_sha256",
    ),
)
def test_resume_rejects_identity_drift(field: str) -> None:
    payload, identity, signature = _payload()
    different = _identity(**{field: "9" * 64})
    assert different != identity
    with pytest.raises(PBDRV5RunContractError, match="identity"):
        validate_rolling_payload(
            payload,
            identity=different,
            epochs=30,
            expected_optimizer_group_signature=signature,
        )


def test_resume_rejects_tensor_tampering() -> None:
    payload, identity, signature = _payload()
    tampered = copy.deepcopy(payload)
    first = next(iter(tampered["state_dict"].values()))
    first.add_(1.0)
    with pytest.raises(PBDRV5RunContractError, match="state SHA"):
        validate_rolling_payload(
            tampered,
            identity=identity,
            epochs=30,
            expected_optimizer_group_signature=signature,
        )


def test_identity_rejects_unknown_arm_and_bad_digest() -> None:
    with pytest.raises(PBDRV5RunContractError, match="arm"):
        _identity(arm="other")
    with pytest.raises(PBDRV5RunContractError, match="SHA-256"):
        _identity(loss_manifest_sha256="bad")
