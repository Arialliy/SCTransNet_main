from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import (
    freeze_tpd_ner_v4_qfg_v2_croa_operational_closure_v2 as freezer,
)


def _publish_fixture_repo(
    tmp_path: Path,
    *,
    missing_source: str | None = None,
    valid_upstream: bool = True,
) -> Path:
    repo = tmp_path / "repo"
    for index, relative in enumerate(freezer.SOURCE_PATHS):
        if relative == missing_source:
            continue
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# fixture operational source {index}\nVALUE = {index}\n",
            encoding="utf-8",
        )
    upstream = repo / freezer.UPSTREAM_LOCK_RELATIVE_PATH
    upstream.parent.mkdir(parents=True, exist_ok=True)
    if valid_upstream:
        upstream.write_bytes(
            (
                freezer.REPO_ROOT
                / freezer.UPSTREAM_LOCK_RELATIVE_PATH
            ).read_bytes()
        )
    else:
        upstream.write_bytes(b"mismatched upstream lock\n")
    return repo


def test_operational_contract_freezes_exactly_four_runtime_sources() -> None:
    assert freezer.SOURCE_PATHS == (
        (
            "experiments/"
            "control_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback.py"
        ),
        (
            "experiments/"
            "publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2.py"
        ),
        (
            "experiments/"
            "generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest.py"
        ),
        (
            "experiments/"
            "generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest_v2.py"
        ),
    )
    assert len(set(freezer.SOURCE_PATHS)) == 4
    assert freezer.UPSTREAM_LOCK_SHA256 == (
        "315f091b75078e65b871946cecae92893e8915bb3951b6fc4dcf3a52c984cbbd"
    )
    assert freezer.RECEIPT_SCHEMA.endswith("fallback_receipt_v1")
    assert freezer.RECEIPT_RELATIVE_PATH.endswith(
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback_receipt.json"
    )
    assert freezer.DEFAULT_OUTPUT_RELATIVE_PATH == (
        "experiments/"
        "tpd_ner_v4_qfg_v2_croa_operational_closure_source_lock_v2.json"
    )


def test_preflight_reports_not_yet_landed_source_without_writing(
    tmp_path: Path,
) -> None:
    missing = freezer.SOURCE_PATHS[-1]
    repo = _publish_fixture_repo(tmp_path, missing_source=missing)
    output = repo / freezer.DEFAULT_OUTPUT_RELATIVE_PATH
    result = freezer.preflight(repo)
    assert result["status"] == "pending"
    assert result["missing_sources"] == [missing]
    assert result["invalid_sources"] == []
    assert result["publish_ready"] is False
    assert result["verify_ready"] is False
    assert result["writes_performed"] is False
    assert result["upstream_posttraining_closure_source_lock"][
        "verified"
    ] is True
    assert not output.exists()


def test_publish_is_canonical_write_once_idempotent_and_live_verifiable(
    tmp_path: Path,
) -> None:
    repo = _publish_fixture_repo(tmp_path)
    receipt = repo / freezer.RECEIPT_RELATIVE_PATH
    output = repo / freezer.DEFAULT_OUTPUT_RELATIVE_PATH
    assert not receipt.exists()

    before = freezer.preflight(repo)
    assert before["status"] == "ready"
    assert before["publish_ready"] is True
    first = freezer.publish(repo)
    assert first["action"] == "publish"
    assert first["writes_performed"] is True
    assert first["verified"] is True
    assert first["source_count"] == 4

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.read_bytes() == freezer.canonical_json_bytes(payload)
    assert payload["schema"] == freezer.LOCK_SCHEMA
    assert payload["source_count"] == 4
    assert tuple(payload["source_sha256"]) == tuple(
        sorted(freezer.SOURCE_PATHS)
    )
    assert payload["upstream_posttraining_closure_source_lock"] == {
        "path": freezer.UPSTREAM_LOCK_RELATIVE_PATH,
        "sha256": freezer.UPSTREAM_LOCK_SHA256,
        "source_count": 15,
    }
    assert payload["receipt_contract"] == {
        "binding_scope": "schema_and_contract_path_only",
        "path": freezer.RECEIPT_RELATIVE_PATH,
        "receipt_content_in_source_lock": False,
        "receipt_may_be_absent_at_freeze_time": True,
        "schema": freezer.RECEIPT_SCHEMA,
    }
    assert not receipt.exists()

    verified = freezer.verify(repo)
    assert verified["action"] == "verify"
    assert verified["output_sha256"] == first["output_sha256"]
    second = freezer.publish(repo)
    assert second["action"] == "verify"
    assert second["writes_performed"] is False
    assert second["output_sha256"] == first["output_sha256"]
    after = freezer.preflight(repo)
    assert after["status"] == "ready"
    assert after["verify_ready"] is True
    assert after["publish_ready"] is False


def test_live_source_change_and_existing_conflict_are_rejected(
    tmp_path: Path,
) -> None:
    repo = _publish_fixture_repo(tmp_path)
    freezer.publish(repo)
    changed = repo / freezer.SOURCE_PATHS[0]
    changed.write_text("changed operational source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from live sources"):
        freezer.verify(repo)
    with pytest.raises(FileExistsError, match="refusing to replace"):
        freezer.publish(repo)
    preflight = freezer.preflight(repo)
    assert preflight["status"] == "blocked"
    assert preflight["output_state"]["content"] == "conflict"


def test_upstream_lock_mismatch_blocks_publish_and_verify(
    tmp_path: Path,
) -> None:
    repo = _publish_fixture_repo(tmp_path, valid_upstream=False)
    preflight = freezer.preflight(repo)
    assert preflight["status"] == "blocked"
    assert preflight["upstream_posttraining_closure_source_lock"][
        "verified"
    ] is False
    with pytest.raises(ValueError, match="upstream 15-source"):
        freezer.publish(repo)


def test_cli_preflight_publish_and_verify_use_temporary_repo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = _publish_fixture_repo(tmp_path)
    output = repo / "locks/operational-v2.json"
    common = ["--repo-root", str(repo), "--output", str(output)]

    assert freezer.main(["--preflight", *common]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["status"] == "ready"
    assert preflight["writes_performed"] is False

    assert freezer.main(["--publish", *common]) == 0
    published = json.loads(capsys.readouterr().out)
    assert published["status"] == "complete"
    assert published["writes_performed"] is True

    assert freezer.main(["--verify", *common]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "complete"
    assert verified["writes_performed"] is False
