from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from experiments.irstd_baseline_teacher import (
    CHECKPOINT_WRAPPER_PREFIX,
    FORMAL_TEACHER,
    OPERATIONAL_BEST,
    IRSTDBaselineTeacherError,
    audit_checkpoint_binding,
    capture_outc_raw_logits,
    declared_baseline_audit_manifest,
    require_teacher_eligible,
    strip_exact_checkpoint_wrapper_prefix,
)
from experiments.irstd_bgcr_run_contract import (
    BGCR_CANDIDATE_KIND,
    BGCR_CANDIDATE_NAME,
    FOLD_ASSIGNMENT_SHA256,
    FOLD_SIZES,
    OFFICIAL_FALSE_FLAGS,
    POOLED_METRIC_SCHEMA,
    SOURCE_SPLIT_MANIFEST_PATH,
    STRATUM_TIE_ORDER,
    IRSTDBGCRRunContractError,
    build_frozen_fold_manifest,
    fold_metric_binding,
    irstd_best_miou_role_key,
    load_train_only_split_manifest,
    pool_oof_sufficient_statistics,
    pool_reference_sufficient_statistics,
    run_contract_manifest,
    select_oof_epoch,
)


def _fold_row(
    fold_index: int,
    epoch: int,
    *,
    intersection_delta: int = 0,
) -> dict[str, object]:
    sample_count = FOLD_SIZES[fold_index]
    intersection = 1_000 + 10 * fold_index + intersection_delta
    union = 2_000 + 100 * fold_index
    false_positive = 400 + fold_index
    row = fold_metric_binding(fold_index, epoch)
    row.update(
        {
            "candidate_kind": BGCR_CANDIDATE_KIND,
            "candidate_name": BGCR_CANDIDATE_NAME,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "matched_target_count": 20 + fold_index,
            "target_count": 22 + fold_index,
            "unmatched_component_pixels": 100 + fold_index,
            "valid_pixel_count": sample_count * 100,
            "matched_tiny_target_count": 5 + fold_index,
            "tiny_target_count": 7 + fold_index,
            "true_positive_pixels": intersection,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": union - intersection - false_positive,
            "niou_sum_numerator": sample_count * 3,
            "niou_sum_denominator": 4,
            "loss_sum_numerator": sample_count,
            "loss_sum_denominator": 4,
        }
    )
    return row


def _epoch_rows(epoch: int, *, intersection_delta: int = 0) -> list[dict[str, object]]:
    return [
        _fold_row(
            fold_index,
            epoch,
            intersection_delta=intersection_delta if fold_index == 0 else 0,
        )
        for fold_index in range(3)
    ]


def test_audited_baseline_ledger_forbids_epoch713_teacher() -> None:
    manifest = declared_baseline_audit_manifest()
    assert manifest["formal_teacher_name"] == FORMAL_TEACHER.name
    assert manifest["operational_reference_name"] == OPERATIONAL_BEST.name
    assert manifest["operational_reference_is_test_selected"] is True
    assert manifest["operational_reference_training_use_prohibited"] is True
    assert OPERATIONAL_BEST.epoch == 713
    assert str(OPERATIONAL_BEST.path) == (
        "/home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_best_mIoU.pth.tar"
    )
    assert OPERATIONAL_BEST.file_bytes == 45_536_899
    assert OPERATIONAL_BEST.file_sha256 == (
        "5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b"
    )
    assert OPERATIONAL_BEST.raw_state_semantic_sha256 == (
        "8d314d45f68de9b6747c5ada4ea7efc4f62a423a4bdf50db2f2d60bd8509d022"
    )
    assert OPERATIONAL_BEST.normalized_state_semantic_sha256 == (
        "5ecf6f812f00e323ab5f8cec55d0ca86ea9f7db2225080bbc0ea947f44e181a4"
    )
    assert OPERATIONAL_BEST.state_key_count == 510
    assert OPERATIONAL_BEST.historical_official_test_selected is True
    assert OPERATIONAL_BEST.teacher_allowed is False
    assert FORMAL_TEACHER.epoch == 1000
    assert str(FORMAL_TEACHER.path) == (
        "/home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_1000.pth.tar"
    )
    assert FORMAL_TEACHER.file_bytes == 45_535_091
    assert FORMAL_TEACHER.file_sha256 == (
        "b4cb66be6e4a410dfd902ba050da82d0b666dd071bfb2c5477a7c3173ff07bc5"
    )
    assert FORMAL_TEACHER.raw_state_semantic_sha256 == (
        "972e7c15f8da8142da85112f535fb555a86293e12d7341d7c5be653fb4076d9b"
    )
    assert FORMAL_TEACHER.normalized_state_semantic_sha256 == (
        "1961ed8ee278fde09508145fe537324172599bfa704c181dc53f756578070b5c"
    )
    assert FORMAL_TEACHER.state_key_count == 510
    assert FORMAL_TEACHER.historical_official_test_selected is False
    assert FORMAL_TEACHER.teacher_allowed is True
    require_teacher_eligible(FORMAL_TEACHER)
    with pytest.raises(IRSTDBaselineTeacherError, match="official-test-selected"):
        require_teacher_eligible(OPERATIONAL_BEST)


def test_audited_checkpoint_files_match_frozen_bytes_and_sha() -> None:
    for binding in (OPERATIONAL_BEST, FORMAL_TEACHER):
        audit = audit_checkpoint_binding(binding, verify_state=False)
        assert Path(audit["path"]) == binding.path
        assert audit["file_verified"] is True
        assert audit["state_verified"] is False
        assert audit["teacher_constructed"] is False


def test_checkpoint_strip_accepts_only_exact_model_wrapper() -> None:
    assert CHECKPOINT_WRAPPER_PREFIX == "model."
    state = {
        "model.weight": torch.ones(1),
        "model.bias": torch.zeros(1),
    }
    stripped = strip_exact_checkpoint_wrapper_prefix(state, expected_key_count=2)
    assert tuple(stripped) == ("weight", "bias")
    with pytest.raises(IRSTDBaselineTeacherError, match="exact 'model.'"):
        strip_exact_checkpoint_wrapper_prefix(
            {"module.weight": torch.ones(1)}, expected_key_count=1
        )
    with pytest.raises(IRSTDBaselineTeacherError, match="key count"):
        strip_exact_checkpoint_wrapper_prefix(state, expected_key_count=3)


class _TinyProbabilityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outc = nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            self.outc.weight.fill_(2.0)
            self.outc.bias.fill_(-0.5)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.outc(image))


def test_outc_hook_returns_raw_logits_and_removes_itself() -> None:
    model = _TinyProbabilityModel().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image = torch.tensor([[[[0.0, 0.5], [1.0, -1.0]]]], dtype=torch.float32)
    raw = capture_outc_raw_logits(model, image)
    assert torch.equal(raw, 2.0 * image - 0.5)
    assert not model.outc._forward_hooks


def test_train_only_manifest_and_frozen_folds_are_complete_and_disjoint() -> None:
    source = load_train_only_split_manifest()
    assert source["source_split"] == "official_train_only"
    assert source["official_test_index_opened"] is False
    assert len(source["official_train_ids"]) == 800
    assert len(source["mask_stats"]) == 800

    manifest = build_frozen_fold_manifest()
    assert manifest["assignment_sha256"] == FOLD_ASSIGNMENT_SHA256
    assert manifest["fold_sizes"] == [267, 267, 266]
    assert manifest["performance_acceptance_margin"] is None
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert manifest[flag] is expected

    validation_sets = [set(fold["validation_ids"]) for fold in manifest["folds"]]
    assert [len(ids) for ids in validation_sets] == [267, 267, 266]
    assert not (validation_sets[0] & validation_sets[1])
    assert not (validation_sets[0] & validation_sets[2])
    assert not (validation_sets[1] & validation_sets[2])
    assert set.union(*validation_sets) == set(source["official_train_ids"])
    for fold, validation_ids in zip(manifest["folds"], validation_sets, strict=True):
        training_ids = set(fold["training_ids"])
        assert not (training_ids & validation_ids)
        assert training_ids | validation_ids == set(source["official_train_ids"])


def test_fold_stratification_and_assignment_hash_are_deterministic() -> None:
    first = build_frozen_fold_manifest()
    second = build_frozen_fold_manifest()
    assert first == second
    assert first["assignment_sha256"] == (
        "a7ce375391e27e53bdad5f67599d470b336f70c304e22a96b8aa3fef6283c583"
    )
    counts = first["stratum_counts_by_fold"]
    assert counts == {
        "empty": [0, 0, 0],
        "tiny_single": [18, 17, 17],
        "tiny_multi": [22, 23, 22],
        "small_non_tiny": [80, 80, 81],
        "larger": [147, 147, 146],
    }
    for stratum in STRATUM_TIE_ORDER:
        assert max(counts[stratum]) - min(counts[stratum]) <= 1


def test_loader_rejects_every_unapproved_manifest_path() -> None:
    with pytest.raises(IRSTDBGCRRunContractError, match="unapproved"):
        load_train_only_split_manifest(Path("/home/ly/SCTransNet_main/other.json"))
    with pytest.raises(IRSTDBGCRRunContractError, match="absolute"):
        load_train_only_split_manifest(Path("relative.json"))
    assert SOURCE_SPLIT_MANIFEST_PATH.is_absolute()


def test_fold_metric_binding_is_train_only_zero_margin() -> None:
    binding = fold_metric_binding(2, 0)
    assert binding["sample_count"] == 266
    assert binding["evaluation_scope"] == "official_train_oof_validation"
    assert binding["performance_acceptance_margin"] is None
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert binding[flag] is expected


def test_pooled_oof_uses_additive_statistics_not_fold_mean() -> None:
    rows = _epoch_rows(0)
    pooled = pool_oof_sufficient_statistics(rows, epoch=0)
    assert pooled["schema"] == POOLED_METRIC_SCHEMA
    assert pooled["sample_count"] == 800
    assert pooled["intersection_pixels"] == 3_030
    assert pooled["union_pixels"] == 6_300
    assert pooled["miou"] == float(Fraction(3_030, 6_300))
    arithmetic_fold_mean = sum(
        Fraction(row["intersection_pixels"], row["union_pixels"]) for row in rows
    ) / 3
    assert Fraction(3_030, 6_300) != arithmetic_fold_mean
    assert pooled["niou"] == 0.75
    assert pooled["test_loss"] == 0.25
    assert pooled["true_positive_pixels"] == 3_030
    assert pooled["false_positive_pixels"] == 1_203
    assert pooled["false_negative_pixels"] == 2_067
    assert pooled["pixel_precision"] == float(Fraction(3_030, 4_233))
    assert pooled["pixel_recall"] == float(Fraction(3_030, 5_097))
    assert pooled["pixel_f1"] == float(Fraction(6_060, 9_330))
    assert pooled["fold_metrics_arithmetic_mean_used"] is False
    assert pooled["exact_counts_pooled_before_ratio"] is True
    assert pooled["performance_acceptance_margin"] is None


def test_reference_pool_reuses_identical_exact_oof_contract() -> None:
    rows = _epoch_rows(0)
    for row in rows:
        row["candidate_kind"] = "frozen_reference"
        row["candidate_name"] = "Current"
    pooled = pool_reference_sufficient_statistics(
        rows,
        reference_name="Current",
    )
    assert pooled["candidate_kind"] == "frozen_reference"
    assert pooled["candidate_name"] == "Current"
    assert pooled["pixel_f1"] == float(Fraction(6_060, 9_330))
    assert irstd_best_miou_role_key(pooled)[0] == Fraction(3_030, 6_300)

    rows[0]["candidate_name"] = "Original"
    with pytest.raises(IRSTDBGCRRunContractError, match="candidate_name"):
        pool_reference_sufficient_statistics(rows, reference_name="Current")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("candidate_kind", "frozen_reference"),
        ("candidate_name", "Current"),
    ),
)
def test_bgcr_pool_rejects_candidate_identity_drift(
    field: str,
    value: str,
) -> None:
    rows = _epoch_rows(0)
    rows[1][field] = value
    with pytest.raises(IRSTDBGCRRunContractError, match=field):
        pool_oof_sufficient_statistics(rows, epoch=0)


def test_pooled_oof_rejects_official_access_or_fold_identity_drift() -> None:
    rows = _epoch_rows(0)
    rows[0]["official_test_accessed"] = True
    with pytest.raises(IRSTDBGCRRunContractError, match="official_test_accessed"):
        pool_oof_sufficient_statistics(rows, epoch=0)

    rows = _epoch_rows(0)
    rows[1]["validation_ids_sha256"] = "0" * 64
    with pytest.raises(IRSTDBGCRRunContractError, match="validation_ids_sha256"):
        pool_oof_sufficient_statistics(rows, epoch=0)


def test_role_key_keeps_exact_ratios_and_has_no_positive_margin() -> None:
    pooled = pool_oof_sufficient_statistics(_epoch_rows(0), epoch=0)
    key = irstd_best_miou_role_key(pooled)
    assert key[0] == Fraction(3_030, 6_300)
    assert key[1] == Fraction(63, 69)
    assert key[2] == -Fraction(303, 80_000)
    assert key[3] == 0.75
    assert key[4] == Fraction(18, 24)
    assert key[5] == -0.25
    assert key[6] == 0
    assert pooled["performance_acceptance_margin"] is None


def test_partial_selector_accepts_any_strict_miou_gain() -> None:
    history = [*_epoch_rows(0), *_epoch_rows(5, intersection_delta=1)]
    selected = select_oof_epoch(history, require_complete=False)
    assert selected["selected_epoch"] == 5
    assert selected["strictly_improves_epoch0_miou"] is True
    assert selected["strictly_improves_epoch0_full_role_key"] is True
    assert selected["performance_acceptance_margin"] is None
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert selected[flag] is expected


def test_epoch_zero_wins_an_exact_pooled_tie() -> None:
    history = [*_epoch_rows(0), *_epoch_rows(5)]
    selected = select_oof_epoch(history, require_complete=False)
    assert selected["selected_epoch"] == 0
    assert selected["strictly_improves_epoch0_full_role_key"] is False
    assert selected["strictly_improves_epoch0_miou"] is False


def test_final_selector_rejects_an_incomplete_history_by_default() -> None:
    with pytest.raises(IRSTDBGCRRunContractError, match="complete OOF history"):
        select_oof_epoch(_epoch_rows(0))


def test_run_manifest_freezes_schedule_role_and_official_boundary() -> None:
    manifest = run_contract_manifest()
    assert manifest["fold_sizes"] == [267, 267, 266]
    assert manifest["fold_seed"] == 42
    assert manifest["train_epochs"] == 120
    assert manifest["evaluate_every"] == 5
    assert manifest["evaluation_epochs"][0] == 0
    assert manifest["evaluation_epochs"][-1] == 120
    assert manifest["performance_acceptance_margin"] is None
    assert manifest["fold_metrics_arithmetic_mean_used"] is False
    assert manifest["reported_pixel_metrics"] == [
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    ]
    for flag, expected in OFFICIAL_FALSE_FLAGS.items():
        assert manifest[flag] is expected
