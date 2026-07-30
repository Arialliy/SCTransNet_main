from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from experiments import (
    freeze_final_model_certification_parent_lock as freezer,
)


def _small_payload() -> dict[str, object]:
    return {
        "schema": freezer.LOCK_SCHEMA,
        "status": "complete",
        "lock_kind": freezer.LOCK_KIND,
        "frozen_model_source_paths": {},
        "upstream_authorities": {},
        "policy": {
            "parent_lock_self_excluded": True,
            "freezer_self_excluded": True,
        },
    }


def test_live_payload_matches_published_lock_and_freezes_parent_semantics() -> None:
    payload = freezer.build_parent_lock_payload()
    output = freezer.DEFAULT_OUTPUT
    observed = json.loads(output.read_text(encoding="utf-8"))

    assert payload == observed
    assert output.read_bytes() == freezer.canonical_json_bytes(payload)
    assert freezer.sha256_file(output) == (
        "42ce912d476bf00e28e2085c12ef5337f012ed9c1ef43082e41bc07d428fb79a"
    )
    assert payload["frozen_model_source_commit"] == (
        "a295f751470c3414bb453d702451cecde41a1524"
    )

    selected = payload["selected_model"]
    assert selected["method_id"] == "d_tss_qfg"
    assert selected["variant"] == "tss_qfg"
    assert selected["operating_point"]["threshold"] == 0.5
    assert selected["operating_point"]["source"] == "fixed_threshold_0_5"
    assert (
        selected["d_training_checkpoint"]["sha256"]
        == freezer.D_CHECKPOINT_SHA256
    )
    assert (
        selected["final_inference_artifact"]["sha256"]
        == freezer.INFERENCE_ARTIFACT_SHA256
    )

    parent = selected["initialization_parent"]
    assert parent["sha256"] == freezer.PARENT_CHECKPOINT_SHA256
    assert parent["usage"] == "initialization_only"
    assert parent["parent_optimizer_inherited"] is False
    assert parent["parent_model_trained_with_child"] is False
    assert parent["child_trainable_scope"] == "all_model_parameters"
    assert parent["children_independent_after_common_initialization"] is True

    shared = payload["arm_training_semantics"]["shared_rule"]
    assert shared["one_independent_child_instance_per_arm"] is True
    assert shared["one_new_optimizer_per_child"] is True
    assert shared["all_child_parameters_trainable"] is True
    assert shared["cross_arm_resume_forbidden"] is True

    claims = payload["claim_boundary"]
    assert claims["final_model_engineering_selected"] is True
    assert claims["paper_core_established"] is False
    assert claims["stability_claim_supported"] is False
    assert claims["official_test_accessed"] is False


def test_payload_uses_only_relative_authority_paths_and_has_no_self_reference() -> None:
    payload = json.loads(freezer.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    serialized = freezer.canonical_json_bytes(payload).decode("utf-8")
    excluded = (
        freezer.DEFAULT_OUTPUT_RELATIVE_PATH,
        "freeze_final_model_certification_parent_lock.py",
        "FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md",
    )
    for value in excluded:
        assert value not in serialized

    assert payload["policy"]["parent_lock_self_excluded"] is True
    assert payload["policy"]["freezer_self_excluded"] is True
    assert payload["policy"]["certification_protocol_self_excluded"] is True
    assert payload["policy"]["certification_commit_bound"] is False
    assert (
        payload["policy"][
            "certification_commit_deferred_to_release_attestation"
        ]
        is True
    )

    paths: list[str] = []
    for record in payload["upstream_authorities"].values():
        paths.append(record["path"])
    selected = payload["selected_model"]
    paths.extend(
        [
            selected["initialization_parent"]["path"],
            selected["d_training_checkpoint"]["path"],
            selected["final_inference_artifact"]["path"],
        ]
    )
    for relative in paths:
        pure = PurePosixPath(relative)
        assert not pure.is_absolute()
        assert ".." not in pure.parts


def test_plan_write_once_verify_and_existing_target_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _small_payload()
    monkeypatch.setattr(
        freezer,
        "build_parent_lock_payload",
        lambda repo_root=freezer.REPO_ROOT: payload,
    )
    output = tmp_path / "parent-lock.json"

    planned = freezer.plan_parent_lock(output=output)
    assert planned["status"] == "ready"
    assert planned["would_write"] is False
    assert not output.exists()

    written = freezer.write_parent_lock_once(output=output)
    assert written["action"] == "write-once"
    assert written["post_write_verified"] is True
    assert output.read_bytes() == freezer.canonical_json_bytes(payload)

    verified = freezer.verify_parent_lock_action(output=output)
    assert verified["action"] == "verify"
    assert verified["verified"] is True
    assert verified["output_sha256"] == written["output_sha256"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freezer.write_parent_lock_once(output=output)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freezer.plan_parent_lock(output=output)


def test_verify_rejects_noncanonical_and_semantically_changed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _small_payload()
    monkeypatch.setattr(
        freezer,
        "build_parent_lock_payload",
        lambda repo_root=freezer.REPO_ROOT: expected,
    )
    output = tmp_path / "parent-lock.json"

    output.write_text(
        json.dumps(expected, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical bytes"):
        freezer.verify_parent_lock(output)

    changed = dict(expected)
    changed["status"] = "changed"
    output.write_bytes(freezer.canonical_json_bytes(changed))
    with pytest.raises(ValueError, match="live payload"):
        freezer.verify_parent_lock(output)


def test_recursive_official_test_boundary_rejects_true_value() -> None:
    freezer._assert_official_test_false(
        {
            "official_test_accessed": False,
            "nested": [{"official_test_claim": False}],
        },
        "fixture",
    )
    with pytest.raises(ValueError, match="official_test_accessed"):
        freezer._assert_official_test_false(
            {"nested": {"official_test_accessed": True}},
            "fixture",
        )


def test_repo_file_rejects_symlink_and_parent_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    regular = root / "regular.json"
    regular.write_text("{}\n", encoding="utf-8")
    linked = root / "linked.json"
    linked.symlink_to(regular)

    assert freezer._repo_file(root, "regular.json", "fixture") == regular
    with pytest.raises(ValueError, match="regular non-symlink"):
        freezer._repo_file(root, "linked.json", "fixture")
    with pytest.raises(ValueError, match="canonical repository-relative"):
        freezer._repo_file(root, "../regular.json", "fixture")


def test_cli_verify_current_parent_lock(
    capsys: pytest.CaptureFixture[str],
) -> None:
    freezer.main(["--verify"])
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == freezer.ACTION_SCHEMA
    assert result["status"] == "complete"
    assert result["action"] == "verify"
    assert result["verified"] is True
    assert result["output_sha256"] == (
        "42ce912d476bf00e28e2085c12ef5337f012ed9c1ef43082e41bc07d428fb79a"
    )


def test_cli_actions_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        freezer.parse_args([])
    with pytest.raises(SystemExit):
        freezer.parse_args(["--plan", "--verify"])

