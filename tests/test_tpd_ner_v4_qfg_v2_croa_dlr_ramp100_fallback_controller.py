from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from experiments import (
    control_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback as controller,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _launcher_text() -> str:
    return """#!/usr/bin/env bash
set -Eeuo pipefail
paired_gpu2_uuid="GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
paired_gpu3_uuid="GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
echo wait_for_gpu_idle=false
exec 9>/tmp/stub-paired.lock
flock -n 9
echo --seed 42 --epochs 800
"""


class StubTerminal:
    def __init__(
        self,
        tmp_path: Path,
        *,
        selected_method: str = "c_qfg_only",
    ) -> None:
        self.root = tmp_path
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.selection = tmp_path / "current/final_selection/selection.json"
        self.selection_markdown = (
            tmp_path / "current/final_selection/selection.md"
        )
        self.artifact = tmp_path / "current/deployment/inference.pth.tar"
        self.manifest = tmp_path / "current/deployment/manifest.json"
        self.closure = tmp_path / "current/closure.json"
        self.launcher = tmp_path / "bin/paired-launcher.sh"
        self.dlr_lock = tmp_path / "locks/dlr-source-lock.json"
        self.dlr_result = tmp_path / "dlr-results"
        self.receipt = tmp_path / "control/receipt.json"
        self.paired_lock = tmp_path / "control/paired.lock"
        self._publish_common()
        self.publish_selection(selected_method)

    def _publish_common(self) -> None:
        closure = {
            "schema": controller.CURRENT_CLOSURE_LOCK_SCHEMA,
            "status": "complete",
            "source_count": 15,
            "source_sha256": {
                f"experiments/stub_{index}.py": f"{index:064x}"
                for index in range(15)
            },
        }
        _write_json(self.closure, closure)
        self.selection_markdown.parent.mkdir(parents=True, exist_ok=True)
        self.selection_markdown.write_text("stub markdown\n", encoding="utf-8")
        self.artifact.parent.mkdir(parents=True, exist_ok=True)
        self.artifact.write_bytes(b"stub inference checkpoint")
        self.launcher.parent.mkdir(parents=True, exist_ok=True)
        self.launcher.write_text(_launcher_text(), encoding="utf-8")
        self.launcher.chmod(0o755)
        dlr_source_lock = {
            "schema": controller.DLR_SOURCE_LOCK_SCHEMA,
            "lock_kind": "training",
            "source_count": 51,
            "source_sha256": {
                f"model/stub_{index}.py": f"{index + 1:064x}"
                for index in range(51)
            },
        }
        _write_json(self.dlr_lock, dlr_source_lock)

    def publish_selection(self, selected_method: str) -> None:
        is_qfg = selected_method in controller.QFG_METHOD_CONTRACT
        if selected_method == "c_qfg_only":
            decision = "SELECT_C_QFG_ONLY"
            variant = "qfg_only"
            training_tss = False
        elif selected_method == "d_tss_qfg":
            decision = "SELECT_D_TSS_QFG"
            variant = "tss_qfg"
            training_tss = True
        elif selected_method == "v4":
            decision = "FALLBACK_TO_FROZEN_V4"
            variant = "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
            training_tss = False
        else:
            raise AssertionError(selected_method)
        assessment = {
            "status": (
                "RELATIVE_IMPROVED" if is_qfg else "DOMINATED"
            ),
            "comparison_method_ids": [
                "baseline",
                "v1",
                "v2",
                "v3",
                "v4",
                "a_control",
                "b_tss",
                "c_qfg_only",
                "d_tss_qfg",
            ],
            "non_isolated_support_count": 2 if is_qfg else 0,
        }
        report = {
            "schema": controller.CURRENT_SELECTION_SCHEMA,
            "status": "complete",
            "dataset": "NUDT-SIRST",
            "training_seed": 42,
            "split_seed": 20260722,
            "official_test_accessed": False,
            "decision": decision,
            "selection": {
                "decision": decision,
                "selected_method_id": selected_method,
                "selected_variant": variant,
                "query_fg_stage_success": is_qfg,
                "final_training_uses_tss": training_tss,
                "final_inference_uses_tss": False,
            },
            "deployment_selection": {
                "selected": {
                    "method_id": selected_method,
                    "variant": variant,
                    "threshold": 0.5,
                },
                "selected_point_is_checkpoint_local": True,
                "cross_checkpoint_metric_stitching": False,
            },
            "candidate_assessments": {
                "c_qfg_only": (
                    assessment
                    if selected_method == "c_qfg_only"
                    else {
                        **assessment,
                        "status": "DOMINATED",
                        "non_isolated_support_count": 0,
                    }
                ),
                "d_tss_qfg": (
                    assessment
                    if selected_method == "d_tss_qfg"
                    else {
                        **assessment,
                        "status": "DOMINATED",
                        "non_isolated_support_count": 0,
                    }
                ),
            },
            "methods": {
                selected_method: {
                    "method_id": selected_method,
                    "variant": variant,
                }
            },
            "query_fg_stage_success": is_qfg,
            "final_model_engineering_selected": is_qfg,
            "final_model_established": is_qfg,
            "final_inference_uses_tss": False,
        }
        _write_json(self.selection, report)
        closure_binding = self.closure_binding()
        manifest = {
            "schema": controller.CURRENT_DEPLOYMENT_MANIFEST_SCHEMA,
            "status": "complete",
            "selected_method_id": selected_method,
            "selected_variant": variant,
            "official_test_accessed": False,
            "final_selection": {
                "path": str(self.selection.resolve()),
                "sha256": _sha(self.selection),
            },
            "posttraining_closure_source_lock": closure_binding,
            "cross_checkpoint_metric_stitching": False,
            "selected_point_is_checkpoint_local": True,
            "export_mode": (
                "strict_head_free_qfg_export"
                if is_qfg
                else "write_once_native_v4_checkpoint_copy"
            ),
        }
        _write_json(self.manifest, manifest)
        self.selected_method = selected_method

    def closure_binding(self) -> dict[str, Any]:
        return {
            "path": str(self.closure.resolve()),
            "sha256": _sha(self.closure),
            "source_count": 15,
            "verified_live": True,
        }

    def closure_loader(
        self,
        path: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert path == self.closure.resolve()
        return json.loads(self.closure.read_text()), self.closure_binding()

    def deployment_validator(self, **_: Any) -> dict[str, Any]:
        return {
            "status": "complete",
            "verified": True,
            "selected_method_id": self.selected_method,
        }

    def markdown_validator(
        self,
        report: dict[str, Any],
        path: Path,
    ) -> None:
        assert report["status"] == "complete"
        assert path == self.selection_markdown.resolve()
        assert path.read_text(encoding="utf-8") == "stub markdown\n"

    def config(self) -> controller.ControllerConfig:
        return controller.ControllerConfig(
            repo_root=self.repo,
            selection=self.selection,
            selection_markdown=self.selection_markdown,
            deployment_artifact=self.artifact,
            deployment_manifest=self.manifest,
            closure_lock=self.closure,
            launcher=self.launcher,
            dlr_source_lock=self.dlr_lock,
            dlr_result_root=self.dlr_result,
            receipt=self.receipt,
            paired_lock=self.paired_lock,
        ).resolved()

    def evaluate(self) -> dict[str, Any]:
        return controller.evaluate_current_terminal(
            self.config(),
            closure_loader=self.closure_loader,
            deployment_validator=self.deployment_validator,
            markdown_validator=self.markdown_validator,
        )


def _paired_stub(config: controller.ControllerConfig) -> dict[str, Any]:
    return {
        "result_root": str(config.dlr_result_root),
        "training_complete": True,
        "formal_epochs": 800,
        "training_seed": 42,
        "split_seed": 20260722,
        "source_lock": {
            "path": str(config.dlr_source_lock),
            "sha256": _sha(config.dlr_source_lock),
            "schema": controller.DLR_SOURCE_LOCK_SCHEMA,
            "source_count": 51,
        },
        "runs": {
            "e_qfg_dlr": {"summary": "e"},
            "f_tss_qfg_dlr": {"summary": "f"},
        },
        "posttraining_selection_complete": False,
        "posttraining_deployment_complete": False,
    }


@pytest.mark.parametrize("method", ["c_qfg_only", "d_tss_qfg"])
def test_frozen_policy_selected_full_qfg_never_launches(
    tmp_path: Path,
    method: str,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method=method)
    evaluation = terminal.evaluate()
    assert evaluation["authoritative_action"] == "no_fallback"
    assert evaluation["paired_required"] is False
    assert evaluation[
        "meaningful_overall_improvement_by_frozen_policy"
    ] is True
    assert evaluation["selected_candidate_status"] == "RELATIVE_IMPROVED"
    assert evaluation["current_terminal"][
        "posttraining_closure_source_lock"
    ]["source_count"] == 15

    calls = []
    receipt = controller.run_worker(
        terminal.config(),
        evaluator=lambda _: copy.deepcopy(evaluation),
        launcher_runner=lambda _: calls.append("launcher") or 0,
    )
    assert calls == []
    assert receipt["status"] == "complete"
    assert receipt["phase"] == "no_fallback"
    assert receipt["official_test_accessed"] is False
    assert receipt["launcher"]["invoked"] is False
    assert receipt["terminal_for_fallback_controller"] is True
    assert receipt["terminal_for_reproducibility_manifest"] is True


def test_v4_authoritative_fallback_launches_once_and_receipts_both_phases(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    evaluation = terminal.evaluate()
    observed_intermediate = []

    def launcher(config: controller.ControllerConfig) -> int:
        receipt = json.loads(config.receipt.read_text(encoding="utf-8"))
        observed_intermediate.append(
            (
                receipt["status"],
                receipt["phase"],
                receipt["launcher"]["invoked"],
                receipt["terminal_for_fallback_controller"],
                receipt["terminal_for_reproducibility_manifest"],
            )
        )
        return 0

    complete = controller.run_worker(
        terminal.config(),
        evaluator=lambda _: copy.deepcopy(evaluation),
        launcher_runner=launcher,
        paired_collector=_paired_stub,
    )
    assert observed_intermediate == [
        ("in_progress", "paired_launching", True, False, False)
    ]
    assert complete["status"] == "complete"
    assert complete["phase"] == "paired_training_complete"
    assert complete["authoritative_action"] == "launch_paired"
    assert complete["paired_required"] is True
    assert complete["paired"]["training_complete"] is True
    assert complete["paired"]["posttraining_selection_complete"] is False
    assert complete["paired"]["posttraining_deployment_complete"] is False
    assert complete["launcher"]["exit_status"] == 0
    assert complete["terminal_for_fallback_controller"] is True
    assert complete["terminal_for_reproducibility_manifest"] is False

    # A second worker validates the live paired bindings and never relaunches.
    calls = []
    again = controller.run_worker(
        terminal.config(),
        evaluator=lambda _: copy.deepcopy(evaluation),
        launcher_runner=lambda _: calls.append("launcher") or 0,
        paired_collector=_paired_stub,
    )
    assert calls == []
    assert again == complete


def test_retryable_launcher_failure_leaves_nonterminal_receipt(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    evaluation = terminal.evaluate()
    with pytest.raises(controller.TerminalPending):
        controller.run_worker(
            terminal.config(),
            evaluator=lambda _: copy.deepcopy(evaluation),
            launcher_runner=lambda _: 75,
            paired_collector=_paired_stub,
        )
    receipt = json.loads(terminal.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "retryable"
    assert receipt["phase"] == "paired_launch_retryable_failure"
    assert receipt["launcher"]["exit_status"] == 75
    assert receipt["terminal_for_fallback_controller"] is False
    assert receipt["terminal_for_reproducibility_manifest"] is False


def test_permanent_launcher_failure_returns_64_on_every_worker(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    evaluation = terminal.evaluate()
    with pytest.raises(controller.ContractError):
        controller.run_worker(
            terminal.config(),
            evaluator=lambda _: copy.deepcopy(evaluation),
            launcher_runner=lambda _: 64,
            paired_collector=_paired_stub,
        )
    receipt = json.loads(terminal.receipt.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["phase"] == "paired_launch_permanent_failure"
    with pytest.raises(controller.ContractError):
        controller.run_worker(
            terminal.config(),
            evaluator=lambda _: copy.deepcopy(evaluation),
            launcher_runner=lambda _: pytest.fail("must not relaunch"),
            paired_collector=_paired_stub,
        )


def test_paired_nonblocking_flock_returns_retryable_busy(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    evaluation = terminal.evaluate()
    terminal.paired_lock.parent.mkdir(parents=True, exist_ok=True)
    with terminal.paired_lock.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(controller.PairedClaimBusy):
            controller.run_worker(
                terminal.config(),
                evaluator=lambda _: copy.deepcopy(evaluation),
                launcher_runner=lambda _: pytest.fail("must not launch"),
                paired_collector=_paired_stub,
            )


def test_missing_any_terminal_artifact_is_exit_75_class(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path)
    terminal.artifact.unlink()
    with pytest.raises(controller.TerminalPending) as error:
        terminal.evaluate()
    assert error.value.exit_code == 75
    assert not terminal.receipt.exists()
    assert not terminal.paired_lock.exists()


def test_qfg_success_requires_explicit_frozen_policy_fields(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="c_qfg_only")
    report = json.loads(terminal.selection.read_text(encoding="utf-8"))
    report["candidate_assessments"]["c_qfg_only"][
        "comparison_method_ids"
    ].remove("baseline")
    _write_json(terminal.selection, report)
    manifest = json.loads(terminal.manifest.read_text(encoding="utf-8"))
    manifest["final_selection"]["sha256"] = _sha(terminal.selection)
    _write_json(terminal.manifest, manifest)
    with pytest.raises(controller.ContractError, match="baseline"):
        terminal.evaluate()


def test_status_is_read_only_and_reports_nonterminal_launch_state(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    evaluation = terminal.evaluate()
    launching = controller._receipt_payload(
        evaluation,
        status="in_progress",
        phase="paired_launching",
        launcher_invoked=True,
        launcher_exit_status=None,
    )
    controller._write_receipt(terminal.receipt, launching)
    before = terminal.receipt.read_bytes()
    status = controller.read_status(
        terminal.config(),
        evaluator=lambda _: copy.deepcopy(evaluation),
    )
    assert status["writes_performed"] is False
    assert status["launcher_invoked"] is False
    assert status["receipt"]["state"] == "paired_launching"
    assert status["receipt"]["fallback_controller_terminal"] is False
    assert status["receipt"]["reproducibility_manifest_terminal"] is False
    assert terminal.receipt.read_bytes() == before


def _publish_paired_run(
    config: controller.ControllerConfig,
    method_id: str,
    spec: dict[str, Any],
) -> None:
    run = config.dlr_result_root / spec["relative_run_directory"]
    run.mkdir(parents=True)
    identity = {
        "variant": spec["variant"],
        "seed": 42,
        "split_seed": 20260722,
        "run_id": (
            "sctransnet:paired:"
            f"NUDT-SIRST:{spec['variant']}:seed-42:split-20260722:"
            f"{spec['run_tag']}"
        ),
        "source_locks": {
            "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock": (
                _sha(config.dlr_source_lock)
            )
        },
    }
    summary = {
        "status": "complete",
        "variant": spec["variant"],
        "candidate_variant": spec["variant"],
        "seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "formal_contract": {"epochs": 800},
        "run_identity": identity,
    }
    _write_json(run / "summary.json", summary)
    (run / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"epoch": epoch}, sort_keys=True) + "\n"
            for epoch in range(1, 801)
        ),
        encoding="utf-8",
    )
    _write_json(run / "exact_journal/active.json", {"epoch": 800})
    for name in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar"):
        (run / name).write_bytes(f"{method_id}:{name}".encode())


def test_cpu_only_paired_closure_collects_two_exact_800_runs(
    tmp_path: Path,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    config = terminal.config()
    for method_id, spec in controller.PAIRED_RUN_SPECS.items():
        _publish_paired_run(config, method_id, spec)
    result = controller.collect_paired_training_closure(config)
    assert result["training_complete"] is True
    assert result["formal_epochs"] == 800
    assert set(result["runs"]) == {"e_qfg_dlr", "f_tss_qfg_dlr"}
    for run in result["runs"].values():
        assert run["formal_epochs_complete"] == 800
        assert set(run["files"]) == {
            "summary",
            "metrics",
            "active_marker",
            "last_checkpoint",
            "pd_primary_checkpoint",
            "miou_secondary_checkpoint",
        }


def test_launcher_runner_forces_exact_paired_roots_and_ignores_stale_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = StubTerminal(tmp_path, selected_method="v4")
    config = terminal.config()
    observed = {}

    class Result:
        returncode = 0

    def fake_run(command: list[str], **kwargs: Any) -> Result:
        observed["command"] = command
        observed.update(kwargs)
        return Result()

    monkeypatch.setenv("TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT", "/wrong")
    monkeypatch.setenv("TPD_NER_DLR_RAMP100_TRAINER", "/wrong/trainer.py")
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    assert controller._run_launcher(config) == 0
    environment = observed["env"]
    assert observed["command"] == [str(config.launcher)]
    assert environment["TPD_NER_DLR_RAMP100_REPO"] == str(config.repo_root)
    assert environment["TPD_NER_DLR_RAMP100_RESULT_ROOT"] == str(
        config.dlr_result_root
    )
    assert environment["TPD_NER_DLR_RAMP100_QFG_OUTPUT_ROOT"] == str(
        config.dlr_result_root / "qfg_dlr_lane"
    )
    assert environment["TPD_NER_DLR_RAMP100_TSS_OUTPUT_ROOT"] == str(
        config.dlr_result_root / "tss_qfg_dlr_lane"
    )
    assert "TPD_NER_DLR_RAMP100_TRAINER" not in environment


def test_public_cli_has_only_0_64_75_operational_exit_semantics(
    tmp_path: Path,
) -> None:
    assert controller.main(["--not-a-mode"]) == 64
    missing = tmp_path / "missing"
    assert (
        controller.main(
            [
                "--dry-run",
                "--repo-root",
                str(tmp_path),
                "--selection",
                str(missing / "selection.json"),
                "--selection-markdown",
                str(missing / "selection.md"),
                "--deployment-artifact",
                str(missing / "artifact.pth.tar"),
                "--deployment-manifest",
                str(missing / "manifest.json"),
                "--closure-source-lock",
                str(missing / "closure.json"),
            ]
        )
        == 75
    )


def test_controller_source_never_installs_or_starts_a_service() -> None:
    source = Path(controller.__file__).read_text(encoding="utf-8")
    assert "systemd-run" not in source
    assert "systemctl" not in source
    assert "nvidia-smi" not in source
    assert "--worker" in source
    assert "--dry-run" in source
    assert "--status" in source
    assert "LOCK_EX | fcntl.LOCK_NB" in source


def test_absolute_path_execution_inserts_repository_root_into_sys_path(
    tmp_path: Path,
) -> None:
    controller_path = Path(controller.__file__).resolve()
    expected_root = str(controller_path.parents[1])
    probe = "\n".join(
        (
            "import json",
            "import runpy",
            "import sys",
            f"scope = runpy.run_path({str(controller_path)!r})",
            "root = str(scope['REPO_ROOT'])",
            "print(json.dumps({",
            "    'root': root,",
            "    'present': root in sys.path,",
            "    'first': sys.path[0],",
            "}))",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    assert observed == {
        "root": expected_root,
        "present": True,
        "first": expected_root,
    }


def test_strict_selection_markdown_restores_formal_method_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import (
        postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selector,
    )

    formal_order = (
        "baseline",
        "v1",
        "v2",
        "v3",
        "v4",
        "a_control",
        "b_tss",
        "c_qfg_only",
        "d_tss_qfg",
    )
    alphabetical_order = tuple(sorted(formal_order))
    report = {
        "status": "complete",
        "methods": {
            method_id: {"method_id": method_id, "marker": index}
            for index, method_id in enumerate(alphabetical_order)
        },
    }
    expected_markdown = "formal-order markdown\n"
    markdown = tmp_path / "selection.md"
    markdown.write_text(expected_markdown, encoding="utf-8")

    def render_markdown(ordered_report: dict[str, Any]) -> str:
        assert tuple(ordered_report["methods"]) == formal_order
        assert {
            method_id: ordered_report["methods"][method_id]
            for method_id in formal_order
        } == {
            method_id: report["methods"][method_id]
            for method_id in formal_order
        }
        return expected_markdown

    monkeypatch.setattr(selector, "render_markdown", render_markdown)
    controller._strict_selection_markdown(report, markdown)
    assert tuple(report["methods"]) == alphabetical_order


def test_strict_selection_markdown_rejects_content_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import (
        postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selector,
    )

    formal_order = (
        "baseline",
        "v1",
        "v2",
        "v3",
        "v4",
        "a_control",
        "b_tss",
        "c_qfg_only",
        "d_tss_qfg",
    )
    report = {
        "methods": {
            method_id: {"method_id": method_id}
            for method_id in sorted(formal_order)
        }
    }
    markdown = tmp_path / "selection.md"
    markdown.write_text("different markdown\n", encoding="utf-8")

    def render_markdown(ordered_report: dict[str, Any]) -> str:
        assert tuple(ordered_report["methods"]) == formal_order
        return "canonical markdown\n"

    monkeypatch.setattr(selector, "render_markdown", render_markdown)
    with pytest.raises(controller.ContractError, match="Markdown conflicts"):
        controller._strict_selection_markdown(report, markdown)
