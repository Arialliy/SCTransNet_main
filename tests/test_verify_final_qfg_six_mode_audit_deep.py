from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from analysis import collect_final_model_validation_statistics as cache_core
from analysis import run_final_qfg_six_mode_audit as audit_runner
from analysis import verify_final_qfg_six_mode_audit_deep as subject


def _sha(character: str) -> str:
    return character * 64


class _DummyLevel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))


class _DummyQFG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [_DummyLevel(value) for value in (0.4, 0.5, 0.6, 0.7)]
        )

    def prepare(self) -> SimpleNamespace:
        gate = torch.tensor(
            [[[-0.6, 0.4], [0.2, 0.8]]],
            dtype=torch.float32,
            device=self.levels[0].alpha.device,
        ).unsqueeze(0)
        return SimpleNamespace(
            levels=tuple(
                SimpleNamespace(
                    gate=gate * float(index + 1) / 4.0,
                    factor=(
                        1.0
                        + torch.tanh(level.alpha)
                        * gate
                        * float(index + 1)
                        / 4.0
                    ),
                )
                for index, level in enumerate(self.levels)
            )
        )


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tpd_qfg = _DummyQFG()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        prepared = self.tpd_qfg.prepare()
        modulation = sum(
            F.interpolate(
                level.factor - 1.0,
                size=image.shape[-2:],
                mode="nearest",
            )
            for level in prepared.levels
        )
        probability = torch.sigmoid(image + 1.5 * modulation)
        return tuple(probability for _ in range(6))


def _context(tmp_path: Path) -> audit_runner.FrozenAuditContext:
    identifiers = tuple(f"deep_{index:03d}" for index in range(133))
    source_lock_sha = _sha("7")
    return audit_runner.FrozenAuditContext(
        repo_root=audit_runner.REPO_ROOT,
        dataset_root=tmp_path,
        validation_ids=identifiers,
        normalization={"mean": 0.0, "std": 1.0},
        checkpoint_sha256=_sha("1"),
        source_checkpoint_sha256=_sha("2"),
        dataset_sha256=_sha("3"),
        evaluator_sha256=_sha("4"),
        normalization_sha256=_sha("5"),
        source_lock_sha256=source_lock_sha,
        validation_ids_sha256=(
            cache_core.validation_identifier_sha256(identifiers)
        ),
        authority_binding={
            "schema": "synthetic_deep_authority_v1",
            "parent_lock": {
                "path": "synthetic/parent.json",
                "sha256": _sha("8"),
                "schema": "synthetic_parent_v1",
            },
            "certification_source_lock": {
                "path": "synthetic/source.json",
                "sha256": source_lock_sha,
                "schema": "synthetic_source_v1",
                "verified": True,
            },
            "official_test_accessed": False,
        },
        live_authority_required=False,
    )


def _loader(
    context: audit_runner.FrozenAuditContext,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    batches = []
    for index, identifier in enumerate(context.validation_ids):
        background = -0.35 + 0.08 * (index % 7)
        image = torch.full(
            (1, 1, 4, 4),
            background,
            dtype=torch.float32,
        )
        row = index % 4
        column = (index // 4) % 4
        image[:, :, row, column] = 1.2 + 0.1 * (index % 3)
        target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        target[:, :, row, column] = 1.0
        batches.append(
            (
                image,
                target,
                torch.tensor([[4, 4]], dtype=torch.int64),
                [identifier],
            )
        )
    return batches


@pytest.fixture(scope="module")
def audit_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, audit_runner.FrozenAuditContext]:
    root = tmp_path_factory.mktemp("deep_audit_source")
    context = _context(root)
    report = audit_runner.execute_six_mode_audit(
        _DummyModel().eval(),
        _loader(context),
        torch.device("cpu"),
        context,
        root / "source_audit",
        bootstrap_replicates=16,
        thresholds_override=(0.0, 0.5, 1.0),
    )
    return report, context


def _copy_and_mutate(
    report: Path,
    destination: Path,
    mutation: str,
) -> Path:
    copied_root = destination / mutation
    shutil.copytree(report.parent, copied_root)
    copied_report = copied_root / report.name
    payload = json.loads(copied_report.read_text(encoding="utf-8"))
    if mutation == "fixed":
        payload["modes"]["full"]["fixed_threshold_metrics"]["miou"] += 0.01
    elif mutation == "budget":
        payload["modes"]["full"]["fa_budget_scan"]["budget_points"][
            "1e-06"
        ]["pd"] -= 0.01
    elif mutation == "component":
        payload["modes"]["level1_off"]["component_difference"][
            "changed_pixel_count"
        ] += 1
    elif mutation == "bootstrap":
        payload["modes"]["level2_off"]["paired_image_bootstrap"][
            "intervals"
        ]["pd"]["lower"] += 0.01
    elif mutation == "factor_region":
        payload["modes"]["level3_off"]["factor_gate_region_statistics"][
            "levels"
        ][0]["regions"]["global"]["factor"]["mean"] += 0.01
    elif mutation == "repeat":
        payload["repeat_inference"]["equivalent"] = not payload[
            "repeat_inference"
        ]["equivalent"]
    else:
        raise AssertionError(mutation)
    copied_report.write_bytes(audit_runner.canonical_json_bytes(payload))
    return copied_report


def test_deep_verifier_recomputes_available_fields_and_states_limits(
    audit_artifact: tuple[Path, audit_runner.FrozenAuditContext],
) -> None:
    report, context = audit_artifact
    result = subject.deep_verify_audit(report, expected_context=context)
    assert result["status"] == "verified"
    assert result["gpu_used"] is False
    assert all(
        value == "fully_recomputed_from_cache"
        for value in result["checks"]["fixed_threshold_metrics"].values()
    )
    assert all(
        record["component_counts"] == "fully_recomputed_from_cache"
        and record["paired_image_bootstrap"] == "fully_recomputed_from_cache"
        for record in result["checks"][
            "counterfactual_cache_derivatives"
        ].values()
    )
    assert (
        result["checks"]["factor_gate_region_statistics"]["status"]
        == "derived_summary_consistency_verified"
    )
    assert (
        result["checks"]["repeat_inference"]["second_repeat_cache_available"]
        is False
    )
    limited_fields = {record["field"] for record in result["limitations"]}
    assert any("repeat_inference" in value for value in limited_fields)
    assert any("factor_gate_region_statistics" in value for value in limited_fields)
    assert result["no_invention_status"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("fixed", "fixed-threshold metrics"),
        ("budget", "selected-point metrics"),
        ("component", "component difference"),
        ("bootstrap", "paired bootstrap"),
        ("factor_region", "factor"),
        ("repeat", "repeat equivalence derivation"),
    ],
)
def test_deep_verifier_rejects_derived_value_tampering(
    audit_artifact: tuple[Path, audit_runner.FrozenAuditContext],
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    report, context = audit_artifact
    tampered = _copy_and_mutate(report, tmp_path, mutation)
    with pytest.raises(subject.DeepAuditVerificationError, match=match):
        subject.deep_verify_audit(tampered, expected_context=context)


def test_formal_fa_grid_and_all_five_budget_optima_are_recomputed(
    audit_artifact: tuple[Path, audit_runner.FrozenAuditContext],
) -> None:
    report, context = audit_artifact
    payload = json.loads(report.read_text(encoding="utf-8"))
    metadata = (
        report.parent
        / payload["modes"]["full"]["cache"]["path"]
    )
    cache = cache_core.load_prediction_cache(
        metadata,
        expected_identity=context.cache_identity("full"),
    )
    scan = subject.recompute_formal_fa_budget_scan(cache)
    assert scan["formal_closed_interval_grid"] is True
    assert set(scan["budget_points"]) == {
        f"{budget:.10g}" for budget in audit_runner.FA_BUDGETS
    }
    assert scan["threshold_count"] == scan["threshold_provenance"][
        "total_unique_threshold_count"
    ]
    for budget in audit_runner.FA_BUDGETS:
        point = scan["budget_points"][f"{budget:.10g}"]
        assert point is not None
        assert point["fa"] <= budget


def test_attestation_is_write_once_and_outside_source_audit(
    audit_artifact: tuple[Path, audit_runner.FrozenAuditContext],
    tmp_path: Path,
) -> None:
    report, context = audit_artifact
    output = tmp_path / "separate_verification" / "deep.json"
    written = subject.write_deep_verification_once(
        report,
        output,
        expected_context=context,
    )
    observed = subject.verify_deep_verification(
        written,
        report,
        expected_context=context,
    )
    assert observed["status"] == "verified"
    with pytest.raises(FileExistsError):
        subject.write_deep_verification_once(
            report,
            output,
            expected_context=context,
        )
    with pytest.raises(
        subject.DeepAuditVerificationError,
        match="outside the source audit directory",
    ):
        subject.write_deep_verification_once(
            report,
            report.parent / "deep.json",
            expected_context=context,
        )
