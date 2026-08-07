from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch

from experiments.irstd_bgcr_run_contract import (
    FOLD_ASSIGNMENT_SHA256,
    OFFICIAL_FALSE_FLAGS,
    OOF_EVALUATION_EPOCHS,
    PROBABILITY_COMPARISON,
    PROBABILITY_THRESHOLD,
    SELECTION_SCHEMA,
    SOURCE_SCOPE,
    SOURCE_SPLIT_MANIFEST_FILE_SHA256,
    canonical_json_sha256,
)
from experiments.pbdr_v4_run_artifacts import exclusive_torch_save, file_sha256
from experiments.pbdr_v4_state_contract import state_semantic_sha256

from model.irstd_core_ring_repair import (
    NEGATIVE_LOGIT_LIMIT,
    POSITIVE_LOGIT_LIMIT,
    PRODUCTION_PARAMETER_COUNT,
    PRODUCTION_PERSISTENT_BUFFER_COUNT,
    PRODUCTION_STATE_KEY_COUNT,
)
from model.tpd8_ner4_qfg2_irstd_crr import (
    CURRENT_INFERENCE_STATE_KEY_COUNT,
    CURRENT_TRAINING_STATE_KEY_COUNT,
    FORMAL_DATASET,
    FORMAL_PARENT_ROLE,
    INTEGRATED_PARAMETER_COUNT,
    INTEGRATED_CANDIDATE_SCHEMA,
    INTEGRATED_STATE_KEY_COUNT,
    IRSTD_CRR_STATE_PREFIX,
    IRSTDBGCRIntegrationError,
    OOF_SELECTOR_SCHEMA,
    STATE_SEMANTIC_HASH_ALGORITHM,
    TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    audit_frozen_current_base,
    build_formal_irstd_bgcr_model,
    load_current_into_frozen_base_strictly,
    load_formal_irstd_bgcr_integrated_candidate,
    strip_current_survival_state_strict,
    validate_formal_irstd_bgcr_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_inference_model,
    build_formal_v4_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
)


@dataclass(slots=True)
class _FormalBundle:
    training_state: dict[str, torch.Tensor]
    inference_state: dict[str, torch.Tensor]
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet
    metadata: dict[str, object]


@pytest.fixture(scope="module")
def formal_bundle() -> _FormalBundle:
    current_training, _ = build_formal_v4_qfg_v2_croa_survival_model(42)
    training_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in current_training.state_dict().items()
    }
    del current_training
    inference_state = strip_current_survival_state_strict(training_state)
    model, metadata = build_formal_irstd_bgcr_model(training_state)
    return _FormalBundle(
        training_state=training_state,
        inference_state=inference_state,
        model=model,
        metadata=metadata,
    )


def test_strict_current_projection_removes_exactly_four_survival_keys(
    formal_bundle: _FormalBundle,
) -> None:
    assert len(SURVIVAL_STATE_KEYS) == 4
    assert len(formal_bundle.training_state) == CURRENT_TRAINING_STATE_KEY_COUNT
    assert len(formal_bundle.inference_state) == CURRENT_INFERENCE_STATE_KEY_COUNT
    assert not set(SURVIVAL_STATE_KEYS) & set(formal_bundle.inference_state)
    assert tuple(formal_bundle.inference_state) == tuple(
        name
        for name in formal_bundle.training_state
        if name not in set(SURVIVAL_STATE_KEYS)
    )


def test_current_projection_rejects_incomplete_and_non_tss_off_states(
    formal_bundle: _FormalBundle,
) -> None:
    incomplete = dict(formal_bundle.training_state)
    del incomplete[SURVIVAL_STATE_KEYS[0]]
    with pytest.raises(IRSTDBGCRIntegrationError, match="568 keys"):
        strip_current_survival_state_strict(incomplete)

    nonzero_survival = dict(formal_bundle.training_state)
    key = SURVIVAL_STATE_KEYS[0]
    nonzero_survival[key] = torch.ones_like(nonzero_survival[key])
    with pytest.raises(IRSTDBGCRIntegrationError, match="TSS-off"):
        strip_current_survival_state_strict(nonzero_survival)


def test_formal_builder_freezes_exact_current_and_exposes_only_repair(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    assert model.bgcr_dataset == FORMAL_DATASET
    assert model.bgcr_parent_role == FORMAL_PARENT_ROLE
    assert len(model.state_dict()) == INTEGRATED_STATE_KEY_COUNT == 595
    assert INTEGRATED_PARAMETER_COUNT == 10_897_350
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        INTEGRATED_PARAMETER_COUNT
    )
    repair_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    assert len(repair_state) == PRODUCTION_STATE_KEY_COUNT == 31
    assert PRODUCTION_PARAMETER_COUNT == 27_220
    assert sum(parameter.numel() for parameter in model.irstd_repair.parameters()) == (
        PRODUCTION_PARAMETER_COUNT
    )
    assert PRODUCTION_PERSISTENT_BUFFER_COUNT == 2
    assert len(tuple(model.irstd_repair.buffers())) == (
        PRODUCTION_PERSISTENT_BUFFER_COUNT
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(name.startswith(IRSTD_CRR_STATE_PREFIX) for name in trainable_names)
    assert {id(parameter) for parameter in model.trainable_parameters()} == {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    assert formal_bundle.metadata[
        "epoch_zero_exact_current_by_zero_terminal_residuals"
    ] is True
    assert formal_bundle.metadata["current_training_state_semantic_sha256"] == (
        state_semantic_sha256(formal_bundle.training_state)
    )
    assert formal_bundle.metadata["current_inference_state_semantic_sha256"] == (
        state_semantic_sha256(formal_bundle.inference_state)
    )
    assert (
        formal_bundle.metadata["current_training_state_semantic_hash_algorithm"]
        == STATE_SEMANTIC_HASH_ALGORITHM
    )
    assert (
        formal_bundle.metadata["current_inference_state_semantic_hash_algorithm"]
        == STATE_SEMANTIC_HASH_ALGORITHM
    )
    assert "current_training_state_sha256" not in formal_bundle.metadata
    assert "current_inference_state_sha256" not in formal_bundle.metadata
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert formal_bundle.metadata[flag] is expected
        assert model.architecture_manifest()[flag] is expected
    assert formal_bundle.metadata["performance_acceptance_margin"] is None


def test_formal_validator_and_training_time_base_audit_replay(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    model.train(True)
    validation = validate_formal_irstd_bgcr_model(
        model,
        expected_current_inference_state=formal_bundle.inference_state,
        require_identity_initialization=True,
    )
    assert validation["current_state_key_count"] == 564
    assert validation["repair_state_key_count"] == 31
    assert validation["integrated_state_key_count"] == 595
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert validation[flag] is expected
    audit = audit_frozen_current_base(model, formal_bundle.inference_state)
    assert audit["all_current_tensors_bitwise_equal"] is True
    assert audit["all_current_parameters_frozen"] is True
    assert audit["all_current_gradients_none"] is True
    assert audit["current_training_modules"] == []
    assert audit["repair_training"] is True


def test_epoch_zero_is_bitwise_current_and_current_graph_is_not_a_raw_parent(
    formal_bundle: _FormalBundle,
) -> None:
    current, _ = build_formal_v4_qfg_v2_croa_inference_model(42)
    current.load_state_dict(formal_bundle.inference_state, strict=True)
    current.eval()
    current.mode = "test"
    model = formal_bundle.model
    model.eval()
    model.mode = "test"
    image = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        expected = current(image)
        observed = model(image)
        context = model._frozen_current_context(image)
    assert torch.equal(observed, expected)
    assert torch.equal(torch.sigmoid(context.out_logits), expected)
    assert torch.count_nonzero(model.irstd_repair.positive_residual_head.weight) == 0
    assert torch.count_nonzero(model.irstd_repair.negative_residual_head.weight) == 0

    with pytest.raises(ValueError, match="already registers tpd_ner"):
        TPD8NER4QFG2IRSTDCRRInferenceSCTransNet(current)


def test_context_api_executes_parent_once_and_cached_path_never_reexecutes_it(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    model.train(True)
    image = torch.randn(1, 1, 32, 32)
    original = model._frozen_current_context
    with patch.object(model, "_frozen_current_context", wraps=original) as parent:
        routing, context = model.forward_for_irstd_training(image)
        assert parent.call_count == 1
        replay = model.forward_repair_from_context(image, context)
        assert parent.call_count == 1
    assert torch.equal(routing.routed_logits, replay.routed_logits)
    assert all(not tensor.requires_grad for tensor in context.tensors())


def test_repair_backward_never_creates_a_current_gradient(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    model.zero_grad(set_to_none=True)
    model.train(True)
    image = torch.randn(1, 1, 32, 32)
    routing, _ = model.forward_for_irstd_training(image)
    loss = (
        routing.routed_logits.square().mean()
        + routing.core_gate_logits.square().mean()
        + routing.halo_gate_logits.square().mean()
    )
    loss.backward()
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(IRSTD_CRR_STATE_PREFIX)
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.irstd_repair.parameters()
    )
    audit = audit_frozen_current_base(model, formal_bundle.inference_state)
    assert audit["all_current_gradients_none"] is True
    model.zero_grad(set_to_none=True)


def test_limit_buffers_are_semantic_and_different_capacity_loads_fail(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    positive_key = f"{IRSTD_CRR_STATE_PREFIX}positive_limit"
    negative_key = f"{IRSTD_CRR_STATE_PREFIX}negative_limit"
    assert float(state[positive_key]) == POSITIVE_LOGIT_LIMIT
    assert float(state[negative_key]) == NEGATIVE_LOGIT_LIMIT
    state[positive_key] = torch.tensor(
        POSITIVE_LOGIT_LIMIT + 1.0,
        dtype=state[positive_key].dtype,
    )
    with pytest.raises(RuntimeError, match="semantic checkpoint mismatch"):
        model.load_state_dict(state, strict=True)
    assert float(model.irstd_repair.positive_limit) == POSITIVE_LOGIT_LIMIT
    assert float(model.irstd_repair.negative_limit) == NEGATIVE_LOGIT_LIMIT


def test_strict_loader_rejects_568_keys_and_dtype_drift_before_install(
    formal_bundle: _FormalBundle,
) -> None:
    model = formal_bundle.model
    with pytest.raises(IRSTDBGCRIntegrationError, match="564 keys"):
        load_current_into_frozen_base_strictly(
            model,
            formal_bundle.training_state,
        )

    wrong_dtype = dict(formal_bundle.inference_state)
    key = next(
        name
        for name, tensor in wrong_dtype.items()
        if tensor.is_floating_point()
    )
    wrong_dtype[key] = wrong_dtype[key].double()
    with pytest.raises(IRSTDBGCRIntegrationError, match="metadata differs"):
        load_current_into_frozen_base_strictly(model, wrong_dtype)
    audit = audit_frozen_current_base(model, formal_bundle.inference_state)
    assert audit["all_current_tensors_bitwise_equal"] is True


def test_formal_seed_is_not_a_runtime_tuning_surface(
    formal_bundle: _FormalBundle,
) -> None:
    with pytest.raises(IRSTDBGCRIntegrationError, match="seed must be 42"):
        build_formal_irstd_bgcr_model(formal_bundle.training_state, seed=7)
    with pytest.raises(
        IRSTDBGCRIntegrationError,
        match="repair_initialization_seed must be 42",
    ):
        build_formal_irstd_bgcr_model(
            formal_bundle.training_state,
            repair_initialization_seed=7,
        )


def _synthetic_candidate_payload(
    formal_bundle: _FormalBundle,
) -> dict[str, object]:
    selected_epoch = 0
    inner_selection: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "dataset": FORMAL_DATASET,
        "role": FORMAL_PARENT_ROLE,
        "selected_epoch": selected_epoch,
        "candidate_epochs": list(OOF_EVALUATION_EPOCHS),
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": (
            SOURCE_SPLIT_MANIFEST_FILE_SHA256
        ),
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }
    selector: dict[str, object] = {
        "schema": OOF_SELECTOR_SCHEMA,
        "dataset": FORMAL_DATASET,
        "role": FORMAL_PARENT_ROLE,
        "source_scope": SOURCE_SCOPE,
        "selected_epoch": selected_epoch,
        "selection": inner_selection,
        "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
        "source_split_manifest_file_sha256": (
            SOURCE_SPLIT_MANIFEST_FILE_SHA256
        ),
        "probability_threshold": PROBABILITY_THRESHOLD,
        "probability_comparison": PROBABILITY_COMPARISON,
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }
    selector["selection_sha256"] = canonical_json_sha256(selector)
    identity: dict[str, object] = {
        "schema": "sctransnet_train_irstd_bgcr_v1/v1/identity",
        "mode": "full",
        "dataset": FORMAL_DATASET,
        "role": FORMAL_PARENT_ROLE,
        "selected_epoch": selected_epoch,
        "current_training_state_tensor_mapping_sha256": "d" * 64,
        "current_training_state_tensor_mapping_hash_algorithm": (
            "tensor_mapping_sha256"
        ),
        "current_inference_state_semantic_sha256": state_semantic_sha256(
            formal_bundle.inference_state
        ),
        "current_inference_state_semantic_hash_algorithm": (
            STATE_SEMANTIC_HASH_ALGORITHM
        ),
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in formal_bundle.model.state_dict().items()
    }
    base_state = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    repair_state = {
        name.removeprefix(IRSTD_CRR_STATE_PREFIX): tensor
        for name, tensor in state.items()
        if name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    return {
        "schema": INTEGRATED_CANDIDATE_SCHEMA,
        "dataset": FORMAL_DATASET,
        "role": FORMAL_PARENT_ROLE,
        "mode": "full",
        "seed": 42,
        "epoch": selected_epoch,
        "oof_selection": selector,
        "oof_selection_file_sha256": "e" * 64,
        "identity": identity,
        "identity_sha256": identity["identity_sha256"],
        "state_dict": state,
        "state_key_count": len(state),
        "state_semantic_sha256": state_semantic_sha256(state),
        "state_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "current_base_state_key_count": len(base_state),
        "current_base_state_semantic_sha256": state_semantic_sha256(base_state),
        "current_base_state_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "repair_state_key_count": len(repair_state),
        "repair_state_semantic_sha256": state_semantic_sha256(repair_state),
        "repair_state_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "architecture_manifest": formal_bundle.model.architecture_manifest(),
        "base_audit": audit_frozen_current_base(
            formal_bundle.model,
            formal_bundle.inference_state,
        ),
        "model_builder_metadata": formal_bundle.metadata,
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }


def test_formal_integrated_candidate_path_loader_is_strict(
    formal_bundle: _FormalBundle,
    tmp_path,
) -> None:
    payload = _synthetic_candidate_payload(formal_bundle)
    path = exclusive_torch_save(tmp_path / "integrated_candidate.pth.tar", payload)
    model, audit = load_formal_irstd_bgcr_integrated_candidate(
        path,
        current_training_state=formal_bundle.training_state,
        expected_file_sha256=file_sha256(path),
    )
    assert type(model) is TPD8NER4QFG2IRSTDCRRInferenceSCTransNet
    assert len(model.state_dict()) == audit["state_key_count"] == 595
    assert audit["current_base_state_key_count"] == 564
    assert audit["repair_state_key_count"] == 31
    assert audit["state_hash_algorithm"] == STATE_SEMANTIC_HASH_ALGORITHM
    for name, expected in payload["state_dict"].items():
        assert torch.equal(model.state_dict()[name].cpu(), expected)
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert audit[flag] is expected


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("official_flag", "official flag differs"),
        ("state_partition", "595 keys"),
        ("selection_schema", "OOF selector schema differs"),
    ),
)
def test_formal_integrated_candidate_rejects_contract_drift_before_load(
    formal_bundle: _FormalBundle,
    mutation: str,
    message: str,
) -> None:
    payload = _synthetic_candidate_payload(formal_bundle)
    if mutation == "official_flag":
        payload["official_test_accessed"] = True
    elif mutation == "state_partition":
        state = payload["state_dict"]
        assert isinstance(state, dict)
        del state[next(iter(state))]
    else:
        selector = payload["oof_selection"]
        assert isinstance(selector, dict)
        selector["schema"] = "wrong"
        selector["selection_sha256"] = canonical_json_sha256(
            {key: value for key, value in selector.items() if key != "selection_sha256"}
        )
    with pytest.raises(IRSTDBGCRIntegrationError, match=message):
        load_formal_irstd_bgcr_integrated_candidate(
            payload,
            current_training_state=formal_bundle.training_state,
        )
