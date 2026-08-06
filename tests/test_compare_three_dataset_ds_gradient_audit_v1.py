from __future__ import annotations

import copy
import math

import pytest

from analysis import compare_three_dataset_ds_gradient_audit_v1 as subject


def _relation(*, pc: bool = False, pa: bool = False, anomaly: bool = False):
    return {
        "persistent_conflict": pc,
        "persistent_alignment": pa,
        "scale_anomaly": anomaly,
        "median_cosine": -0.2 if pc else (0.3 if pa else 0.0),
        "median_norm_ratio_to_final": 0.5,
    }


def _available_stratum():
    groups = {}
    for group in subject.SHARED_GROUPS:
        groups[group] = {
            "heads": {head: _relation() for head in subject.AUXILIARY_HEADS},
            "auxiliary": {"aggregate_conflict": False},
        }
    return {
        "available": True,
        "groups": groups,
        "batch_contract_sha256": "c" * 64,
    }


def _units():
    ready = {}
    for dataset_index, dataset in enumerate(subject.DATASETS):
        ready[dataset] = {}
        for role_index, role in enumerate(subject.CHECKPOINT_ROLES):
            ready[dataset][role] = {
                "manifest_sha256": "a" * 64,
                "checkpoint_sha256": f"{dataset_index}{role_index}".ljust(64, "b"),
                "training_state_dict_sha256": f"{role_index}{dataset_index}".ljust(
                    64, "c"
                ),
                "parameter_partition_sha256": "d" * 64,
                "engineering_valid": True,
                "engineering_failure_reasons": [],
                "strata": {
                    "tiny_positive": _available_stratum(),
                    "normal_positive": _available_stratum(),
                    "background_only": {
                        "available": dataset != "NUDT-SIRST",
                        "batch_contract_sha256": "e" * 64,
                        **(
                            {"groups": _available_stratum()["groups"]}
                            if dataset != "NUDT-SIRST"
                            else {}
                        ),
                    },
                },
            }
    return ready


def _set_signature(
    units,
    *,
    dataset: str,
    role: str,
    pc: bool = False,
    pa: bool = False,
    ac: bool = False,
    anomaly: bool = False,
    stratum: str = "tiny_positive",
    group: str = "encoder_shared",
    head: str = "gt5",
):
    group_row = units[dataset][role]["strata"][stratum]["groups"][group]
    group_row["heads"][head] = _relation(pc=pc, pa=pa, anomaly=anomaly)
    group_row["auxiliary"]["aggregate_conflict"] = ac


def test_pc_and_pa_are_recomputed_at_frozen_thresholds():
    pc = subject.classify_head_relation(
        [-0.3, -0.2, -0.1, 0.01],
        [0.25, 0.3, 0.4, 0.5],
    )
    assert pc["persistent_conflict"] is True
    assert pc["persistent_alignment"] is False

    pa = subject.classify_head_relation(
        [0.1, 0.2, 0.3, 0.4],
        [0.25, 0.3, 0.4, 0.5],
    )
    assert pa["persistent_conflict"] is False
    assert pa["persistent_alignment"] is True


def test_ac_requires_direction_frequency_and_material_scale():
    passed = subject.classify_auxiliary_relation(
        [-0.4, -0.2, -0.1, 0.1],
        [1.5, 1.6, 2.0, 3.0],
    )
    assert passed["aggregate_conflict"] is True
    failed_scale = subject.classify_auxiliary_relation(
        [-0.4, -0.2, -0.1, 0.1],
        [0.5, 0.6, 0.7, 0.8],
    )
    assert failed_scale["aggregate_conflict"] is False


def test_two_of_three_paired_roles_authorize_design_but_never_training():
    units = _units()
    for dataset in subject.DATASETS[:2]:
        for role in subject.CHECKPOINT_ROLES:
            _set_signature(units, dataset=dataset, role=role, pc=True, ac=True)
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_AUTHORIZE
    assert result["ds_v2_design_authorized"] is True
    assert result["ds_v2_training_authorized"] is False
    assert result["tiny_gradient_conflict_supported"] is True


def test_conditional_background_requires_both_available_datasets():
    units = _units()
    for role in subject.CHECKPOINT_ROLES:
        _set_signature(
            units,
            dataset="NUAA-SIRST",
            role=role,
            pc=True,
            ac=True,
            stratum="background_only",
        )
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_NO_CONFLICT
    row = next(
        value
        for value in result["signatures"]
        if value["head"] == "gt5"
        and value["group"] == "encoder_shared"
        and value["stratum"] == "background_only"
    )
    assert row["available_dataset_count"] == 2
    assert row["required_dataset_count_k_s"] == 2
    assert row["paired_pc_ac_pass_count"] == 1


def test_domain_reversal_blocks_global_reweighting():
    units = _units()
    _set_signature(
        units,
        dataset="NUAA-SIRST",
        role="best_miou",
        pc=True,
        ac=True,
    )
    _set_signature(
        units,
        dataset="NUDT-SIRST",
        role="best_miou",
        pa=True,
    )
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_DOMAIN_REVERSAL
    assert result["ds_v2_design_authorized"] is False


def test_engineering_failure_has_highest_precedence():
    units = _units()
    units["NUAA-SIRST"]["best_miou"] = copy.deepcopy(
        units["NUAA-SIRST"]["best_miou"]
    )
    units["NUAA-SIRST"]["best_miou"]["engineering_valid"] = False
    units["NUAA-SIRST"]["best_miou"]["engineering_failure_reasons"] = [
        "sentinel_replay"
    ]
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_ENGINEERING_INVALID
    assert result["ds_v2_training_authorized"] is False


def test_scale_anomaly_precedes_no_conflict():
    units = _units()
    _set_signature(
        units,
        dataset="NUAA-SIRST",
        role="best_miou",
        anomaly=True,
    )
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_SCALE_ANOMALY
    assert result["ds_v2_design_authorized"] is False


def test_scale_anomaly_vetoes_an_otherwise_authorized_signature():
    units = _units()
    for dataset in subject.DATASETS[:2]:
        for role in subject.CHECKPOINT_ROLES:
            _set_signature(units, dataset=dataset, role=role, pc=True, ac=True)
    _set_signature(
        units,
        dataset="IRSTD-1K",
        role="best_pd",
        anomaly=True,
        group="ner_shared",
        head="gt2",
    )
    result = subject.decide_from_units(units)
    assert result["authorized_signatures"]
    assert result["decision"] == subject.DECISION_SCALE_ANOMALY
    assert result["ds_v2_design_authorized"] is False
    assert result["tiny_gradient_conflict_supported"] is False


def test_best_pd_domain_reversal_is_not_ignored():
    units = _units()
    _set_signature(
        units,
        dataset="NUAA-SIRST",
        role="best_pd",
        pc=True,
        ac=True,
    )
    _set_signature(
        units,
        dataset="NUDT-SIRST",
        role="best_pd",
        pa=True,
    )
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_DOMAIN_REVERSAL


def _serialized_group(gram, *, parameter_numel=100):
    seed = {
        "gram_head_order": list(subject.HEAD_ORDER),
        "gram_6x6": copy.deepcopy(gram),
        "parameter_numel": parameter_numel,
        "heads": {head: {} for head in subject.HEAD_ORDER},
        "aux_total": {},
    }
    # Populate the redundant serialized fields from the same sole source, then
    # pass through the strict validator once more in the actual assertion.
    norms = [math.sqrt(gram[index][index]) for index in range(6)]
    final_norm = norms[-1]
    for index, head in enumerate(subject.HEAD_ORDER):
        dot = gram[index][5]
        seed["heads"][head] = {
            "raw_l2_norm": norms[index],
            "gradient_rms": norms[index] / math.sqrt(parameter_numel),
            "norm_ratio_to_final": None if final_norm == 0 else norms[index] / final_norm,
            "cosine_to_final": subject._safe_cosine(dot, norms[index], final_norm),
            "dot_with_final": dot,
            "projection_onto_final": None if final_norm == 0 else dot / (final_norm**2),
        }
    aux_square = sum(gram[left][right] for left in range(5) for right in range(5))
    aux_dot = sum(gram[index][5] for index in range(5))
    total_square = aux_square + gram[5][5] + 2 * aux_dot
    aux_norm = math.sqrt(max(0.0, aux_square))
    total_norm = math.sqrt(max(0.0, total_square))
    norm_sum = sum(norms)
    seed["aux_total"] = {
        "aux_l2_norm": aux_norm,
        "final_l2_norm": final_norm,
        "total_l2_norm": total_norm,
        "aux_final_dot": aux_dot,
        "cosine_aux_final": subject._safe_cosine(aux_dot, aux_norm, final_norm),
        "aux_to_final_norm_ratio": None if final_norm == 0 else aux_norm / final_norm,
        "cancellation": None if norm_sum == 0 else total_norm / norm_sum,
        "individual_head_norm_sum": norm_sum,
    }
    return seed


def _diagonal_gram(values=(1.0, 4.0, 9.0, 16.0, 25.0, 36.0)):
    return [
        [float(values[row]) if row == column else 0.0 for column in range(6)]
        for row in range(6)
    ]


def test_gram_is_the_sole_source_and_all_derived_fields_are_checked():
    serialized = _serialized_group(_diagonal_gram())
    recomputed = subject._validate_and_recompute_group(
        serialized,
        expected_parameter_numel=100,
        label="unit",
    )
    assert recomputed["heads"]["gt3"]["raw_l2_norm"] == 3.0
    assert recomputed["aux_total"]["aux_l2_norm"] == pytest.approx(math.sqrt(55.0))

    corrupted = copy.deepcopy(serialized)
    corrupted["heads"]["gt3"]["cosine_to_final"] = 0.25
    with pytest.raises(subject.DSGAComparisonError, match="differs from gram_6x6"):
        subject._validate_and_recompute_group(
            corrupted,
            expected_parameter_numel=100,
            label="unit",
        )


def test_gram_rejects_asymmetry_nonfinite_and_negative_diagonal():
    asymmetric = _serialized_group(_diagonal_gram())
    asymmetric["gram_6x6"][0][1] = 0.5
    with pytest.raises(subject.DSGAComparisonError, match="not symmetric"):
        subject._validate_and_recompute_group(
            asymmetric, expected_parameter_numel=100, label="unit"
        )

    nonfinite = _serialized_group(_diagonal_gram())
    nonfinite["gram_6x6"][0][0] = float("nan")
    with pytest.raises(subject.DSGAComparisonError, match="must be finite"):
        subject._validate_and_recompute_group(
            nonfinite, expected_parameter_numel=100, label="unit"
        )

    negative = _serialized_group(_diagonal_gram())
    negative["gram_6x6"][0][0] = -1.0
    with pytest.raises(subject.DSGAComparisonError, match="diagonal is negative"):
        subject._validate_and_recompute_group(
            negative, expected_parameter_numel=100, label="unit"
        )


def test_zero_final_gradient_keeps_null_ratios_and_cannot_trigger_pc_or_ac():
    gram = _diagonal_gram(values=(1.0, 4.0, 9.0, 16.0, 25.0, 0.0))
    serialized = _serialized_group(gram)
    recomputed = subject._validate_and_recompute_group(
        serialized, expected_parameter_numel=100, label="unit"
    )
    assert recomputed["heads"]["gt5"]["norm_ratio_to_final"] is None
    assert recomputed["aux_total"]["aux_to_final_norm_ratio"] is None
    head = subject.classify_head_relation([None] * 4, [None] * 4)
    auxiliary = subject.classify_auxiliary_relation([None] * 4, [None] * 4)
    assert head["persistent_conflict"] is False
    assert head["persistent_alignment"] is False
    assert auxiliary["aggregate_conflict"] is False


def test_scale_anomaly_ignores_batches_with_invalid_final_norm():
    assert subject._scale_anomaly_batch_indices(
        [5000.0, 1500.0, 1200.0, None],
        [0.0, subject.FINAL_NORM_MIN / 2, 1.0, 1.0],
    ) == [2]


def test_nested_checkpoint_role_and_sha_are_strict():
    valid = {
        "checkpoint": {"role": "best_miou", "sha256": "a" * 64},
        "training_state_dict_sha256": "b" * 64,
    }
    assert subject._validate_checkpoint_binding(
        valid, expected_role="best_miou"
    ) == ("a" * 64, "b" * 64)

    missing_nested_role = copy.deepcopy(valid)
    del missing_nested_role["checkpoint"]["role"]
    with pytest.raises(subject.DSGAComparisonError, match="nested checkpoint role"):
        subject._validate_checkpoint_binding(
            missing_nested_role, expected_role="best_miou"
        )

    wrong_nested_role = copy.deepcopy(valid)
    wrong_nested_role["checkpoint"]["role"] = "best_pd"
    with pytest.raises(subject.DSGAComparisonError, match="nested checkpoint role"):
        subject._validate_checkpoint_binding(wrong_nested_role, expected_role="best_miou")


def _partition_group(names, numel):
    return {
        "parameter_tensor_count": len(names),
        "parameter_numel": numel,
        "ordered_parameter_names": list(names),
        "ordered_parameter_names_sha256": subject.canonical_sha256(list(names)),
    }


def _minimal_partition():
    shared_names = [f"shared_{index}.weight" for index in range(4)]
    local_names = [f"local_{head}.weight" for head in subject.HEAD_ORDER]
    all_names = shared_names + local_names
    return {
        "atomic_groups": {"all": _partition_group(all_names, len(all_names))},
        "shared_groups": {
            group: _partition_group([shared_names[index]], 1)
            for index, group in enumerate(subject.SHARED_GROUPS)
        },
        "head_local_groups": {
            head: _partition_group([local_names[index]], 1)
            for index, head in enumerate(subject.HEAD_ORDER)
        },
        "trainable_parameter_tensor_count": len(all_names),
        "trainable_parameter_numel": len(all_names),
        "all_trainable_parameters_assigned_once": True,
        "shared_groups_mutually_exclusive": True,
    }


def test_parameter_partition_names_counts_numel_and_sha_are_recomputed():
    ready = subject._validate_parameter_partition(_minimal_partition())
    assert subject._is_sha256(ready["sha256"])
    assert set(ready["shared_group_numel"]) == set(subject.SHARED_GROUPS)

    corrupted = _minimal_partition()
    corrupted["shared_groups"]["encoder_shared"][
        "ordered_parameter_names_sha256"
    ] = "0" * 64
    with pytest.raises(subject.DSGAComparisonError, match="not reproducible"):
        subject._validate_parameter_partition(corrupted)


def _raw_available_stratum():
    batches = []
    for batch_index in range(4):
        batches.append(
            {
                "batch_index": batch_index,
                "forward_seed": batch_index + 1,
                "sample_ids": [f"NUAA-SIRST::id_{batch_index}_{i}" for i in range(16)],
                "images_sha256": f"{batch_index + 1:x}" * 64,
                "masks_sha256": f"{batch_index + 5:x}" * 64,
                "shared_groups": {
                    group: _serialized_group(_diagonal_gram(), parameter_numel=100)
                    for group in subject.SHARED_GROUPS
                },
            }
        )
    return {
        "required_or_conditional": "required",
        "available": True,
        "sample_count": 64,
        "distinct_source_count": 24,
        "diversity_target": 24,
        "max_repeat_cap": 3,
        "diversity_target_limited_by_natural_availability": False,
        "batches": batches,
    }


def test_available_stratum_requires_exact_batches_of_16_and_builds_replay_hash():
    raw = _raw_available_stratum()
    ready = subject._extract_available_stratum(
        raw,
        dataset="NUAA-SIRST",
        role="best_miou",
        stratum="tiny_positive",
        shared_group_numel={group: 100 for group in subject.SHARED_GROUPS},
    )
    assert subject._is_sha256(ready["batch_contract_sha256"])
    assert all(len(batch["sample_ids"]) == 16 for batch in ready["batch_contracts"])

    raw["batches"][0]["sample_ids"].pop()
    with pytest.raises(subject.DSGAComparisonError, match="16 sample IDs"):
        subject._extract_available_stratum(
            raw,
            dataset="NUAA-SIRST",
            role="best_miou",
            stratum="tiny_positive",
            shared_group_numel={group: 100 for group in subject.SHARED_GROUPS},
        )


def test_batch_or_parameter_partition_mismatch_is_engineering_invalid():
    units = _units()
    units["NUAA-SIRST"]["best_pd"]["strata"]["tiny_positive"][
        "batch_contract_sha256"
    ] = "f" * 64
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_ENGINEERING_INVALID
    assert any(
        row.get("reason") == "checkpoint_roles_do_not_reuse_identical_batches"
        for row in result["engineering_failures"]
    )

    units = _units()
    units["IRSTD-1K"]["best_pd"]["parameter_partition_sha256"] = "f" * 64
    result = subject.decide_from_units(units)
    assert result["decision"] == subject.DECISION_ENGINEERING_INVALID
    assert any(
        row.get("reason") == "six_roles_do_not_share_one_parameter_partition"
        for row in result["engineering_failures"]
    )
