from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from experiments import (
    auto_seal_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest as auto,
)
from experiments import (
    generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest as generator,
)


def _receipt(
    *,
    status: str,
    phase: str,
) -> dict[str, object]:
    paired_terminal = phase == "paired_training_complete"
    current_terminal = phase == "no_fallback"
    return {
        "schema": generator.CONTROLLER_SCHEMA,
        "status": status,
        "phase": phase,
        "official_test_accessed": False,
        "authoritative_action": (
            "launch_paired" if paired_terminal or status != "complete"
            else "no_fallback"
        ),
        "selected_method_id": "v4" if paired_terminal else "c_qfg_only",
        "selected_variant": "fixture",
        "selected_candidate_status": (
            None if paired_terminal else "RELATIVE_IMPROVED"
        ),
        "decision": "fixture",
        "query_fg_stage_success": current_terminal,
        "final_model_engineering_selected": current_terminal,
        "final_model_established": current_terminal,
        "meaningful_overall_improvement_by_frozen_policy": current_terminal,
        "meaningful_improvement_basis": "fixture",
        "paired_required": paired_terminal or status != "complete",
        "current_terminal": {},
        "launcher": {},
        "paired": {
            "required": paired_terminal or status != "complete",
            "training_complete": paired_terminal,
            "posttraining_selection_complete": False,
            "posttraining_deployment_complete": False,
            "runs": {},
        },
        "terminal_for_fallback_controller": status == "complete",
        "terminal_for_reproducibility_manifest": current_terminal,
        "receipt_write_policy": "atomic_state_transition",
    }


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(auto._canonical_bytes(payload))


def _layout(tmp_path: Path) -> generator.EvidenceLayout:
    return generator.default_layout(
        repo_root=tmp_path,
        output_dir=tmp_path / "sealed",
        enforce_frozen_lock_digests=False,
    )


def test_missing_receipt_returns_75_without_invoking_generator(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with mock.patch.object(generator, "execute") as execute:
        assert auto.main(["--worker", "--receipt", str(missing)]) == 75
    execute.assert_not_called()
    assert not (tmp_path / "sealed").exists()


def test_in_progress_receipt_returns_75_without_writes(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    _write_receipt(
        receipt,
        _receipt(status="in_progress", phase="paired_launching"),
    )
    with mock.patch.object(generator, "execute") as execute:
        assert auto.main(["--worker", "--receipt", str(receipt)]) == 75
    execute.assert_not_called()


def test_no_fallback_dry_run_routes_current_and_only_preflights(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    _write_receipt(
        receipt,
        _receipt(status="complete", phase="no_fallback"),
    )
    layout = _layout(tmp_path)
    calls: list[dict[str, object]] = []

    def executor(
        observed_layout: generator.EvidenceLayout,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"layout": observed_layout, **kwargs})
        return {
            "status": "ready",
            "action": "preflight",
            "writes_performed": False,
        }

    result = auto.auto_seal(
        receipt_path=receipt,
        worker=False,
        layout_factory=lambda **_: layout,
        executor=executor,
    )
    assert result["mode"] == "dry_run"
    assert result["terminal_family"] == "current"
    assert result["seal_requested"] is False
    assert result["writes_performed"] is False
    assert calls[0]["terminal_family"] == "current"
    assert calls[0]["preflight"] is True
    assert calls[0]["verify"] is False


def test_paired_complete_worker_routes_paired_and_may_seal(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    _write_receipt(
        receipt,
        _receipt(status="complete", phase="paired_training_complete"),
    )
    layout = _layout(tmp_path)
    calls: list[dict[str, object]] = []

    def executor(
        observed_layout: generator.EvidenceLayout,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"layout": observed_layout, **kwargs})
        return {
            "status": "complete",
            "action": "publish",
            "writes_performed": True,
        }

    result = auto.auto_seal(
        receipt_path=receipt,
        worker=True,
        layout_factory=lambda **_: layout,
        executor=executor,
    )
    assert result["mode"] == "worker"
    assert result["terminal_family"] == "paired"
    assert result["seal_requested"] is True
    assert result["writes_performed"] is True
    assert calls[0]["terminal_family"] == "paired"
    assert calls[0]["preflight"] is False
    assert calls[0]["verify"] is False


def test_invalid_complete_phase_returns_64_without_generator_call(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    _write_receipt(
        receipt,
        _receipt(status="complete", phase="paired_launching"),
    )
    with mock.patch.object(generator, "execute") as execute:
        assert auto.main(["--dry-run", "--receipt", str(receipt)]) == 64
    execute.assert_not_called()


def test_noncanonical_or_conflicting_receipt_returns_64(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    payload = _receipt(status="complete", phase="no_fallback")
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    assert auto.main(["--dry-run", "--receipt", str(receipt)]) == 64

    payload["unexpected"] = True
    _write_receipt(receipt, payload)
    assert auto.main(["--dry-run", "--receipt", str(receipt)]) == 64
