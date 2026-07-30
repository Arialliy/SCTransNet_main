from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from analysis import collect_final_model_validation_statistics as cache_core
from analysis import run_final_qfg_six_mode_audit as subject


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
        self.forward_contracts: list[tuple[bool, bool]] = []

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self.forward_contracts.append(
            (torch.is_inference_mode_enabled(), self.training)
        )
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


def _synthetic_context(tmp_path: Path) -> subject.FrozenAuditContext:
    identifiers = tuple(f"synthetic_{index:03d}" for index in range(133))
    source_lock_sha = _sha("7")
    parent_lock_sha = _sha("8")
    return subject.FrozenAuditContext(
        repo_root=subject.REPO_ROOT,
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
            "schema": "synthetic_f1_authority_v1",
            "parent_lock": {
                "path": "synthetic/parent.json",
                "sha256": parent_lock_sha,
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


def _synthetic_loader(
    context: subject.FrozenAuditContext,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    base = torch.full((1, 1, 4, 4), -0.15, dtype=torch.float32)
    base[:, :, 0, 0] = 1.5
    target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target[:, :, 0, 0] = 1.0
    return [
        (
            base.clone(),
            target.clone(),
            torch.tensor([[4, 4]], dtype=torch.int64),
            [identifier],
        )
        for identifier in context.validation_ids
    ]


def test_live_preflight_binds_frozen_parent_and_defers_only_missing_source_lock() -> None:
    result = subject.preflight()
    assert result["gpu_used"] is False
    assert result["writes_performed"] is False
    assert result["validation_count"] == 133
    assert (
        result["final_inference_artifact_sha256"]
        == subject.EXPECTED_INFERENCE_SHA256
    )
    assert result["modes"] == list(subject.PUBLIC_MODES)
    source = result["certification_source_lock"]
    if source["verified"]:
        assert result["status"] == "ready"
        assert result["cache_identity_ready"] is True
        assert len(source["sha256"]) == 64
    else:
        assert result["status"] == "source_lock_pending"
        assert result["cache_identity_ready"] is False
        assert source["status"] == "deferred_missing"


def test_synthetic_six_mode_audit_is_executable_and_self_verifying(
    tmp_path: Path,
) -> None:
    context = _synthetic_context(tmp_path)
    model = _DummyModel().eval()
    report_path = subject.execute_six_mode_audit(
        model,
        _synthetic_loader(context),
        torch.device("cpu"),
        context,
        tmp_path / "published_audit",
        bootstrap_replicates=16,
        thresholds_override=(0.0, 0.5, 1.0),
    )
    report = subject.verify_audit_report(
        report_path,
        expected_context=context,
    )

    assert set(report["modes"]) == set(subject.PUBLIC_MODES)
    assert report["execution_contract"]["modes"] == list(subject.PUBLIC_MODES)
    assert report["repeat_inference"]["equivalent"] is True
    assert report["repeat_inference"]["same_cache_content_sha256"] is True
    assert (
        report["functional_gate"]["full_vs_qfg_off_functionally_different"]
        is True
    )
    assert report["functional_gate"]["nontrivial_factor_use"] is True
    assert report["functional_gate"]["qfg_functionally_active"] is True
    assert (
        report["execution_contract"]["certification_source_lock_sha256"]
        == context.source_lock_sha256
    )
    assert all(enabled and not training for enabled, training in model.forward_contracts)
    assert len(model.forward_contracts) == 7 * 133
    for mode in subject.PUBLIC_MODES:
        record = report["modes"][mode]
        assert record["factor_gate_region_statistics"]["status"] == "complete"
        assert set(record["fa_budget_scan"]["budget_points"]) == {
            f"{budget:.10g}" for budget in subject.FA_BUDGETS
        }
    for mode in subject.COUNTERFACTUAL_MODES:
        assert (
            report["modes"][mode]["paired_image_bootstrap"]["replicates"]
            == 16
        )
        assert report["modes"][mode]["component_difference"]["status"] == "complete"


def test_failed_staging_leaves_output_path_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "audit"

    def fail(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(subject, "_execute_six_mode_audit_staged", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        subject.execute_six_mode_audit(
            nn.Module(),
            (),
            torch.device("cpu"),
            _synthetic_context(tmp_path),
            output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".audit.staging-*"))


def test_report_symlink_is_rejected_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "report.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        subject.verify_audit_report(
            link,
            expected_context=_synthetic_context(tmp_path),
        )
