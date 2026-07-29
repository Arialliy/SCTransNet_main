from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from experiments import (
    generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest as manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_json(path: Path, payload: object) -> None:
    _write(path, manifest._canonical_json_bytes(payload))


def _source_map(
    root: Path,
    *,
    prefix: str,
    count: int,
    mainline: bool,
) -> dict[str, str]:
    relative_paths: list[str] = []
    if mainline:
        relative_paths.extend(relative for _, relative in manifest.MAINLINE_SOURCES)
    index = 0
    while len(relative_paths) < count:
        relative_paths.append(f"fixture_sources/{prefix}/source_{index:03d}.txt")
        index += 1
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.exists():
            _write(path, f"{relative}\n".encode())
        result[relative] = _sha(path)
    return result


def _make_locks(
    layout: manifest.EvidenceLayout,
) -> dict[str, str]:
    root = layout.repo_root
    _write(layout.parent_checkpoint, b"fixture-parent-checkpoint")
    parent = {
        "epoch": 489,
        "path": layout.parent_checkpoint.relative_to(root).as_posix(),
        "role": "best_validation_miou_secondary",
        "sha256": _sha(layout.parent_checkpoint),
        "state_dict_sha256": "a" * 64,
        "variant": "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on",
    }
    lock_by_name = {spec.name: spec for spec in layout.locks}
    current_training = {
        "schema": "fixture_current_training",
        "source_count": 48,
        "source_sha256": _source_map(
            root,
            prefix="current_training",
            count=48,
            mainline=True,
        ),
        "formal_contract": {"official_test_accessed": False},
        "parent_checkpoint": parent,
    }
    _write_json(
        lock_by_name["current_training_48"].path,
        current_training,
    )
    current_training_sha = _sha(
        lock_by_name["current_training_48"].path
    )

    current_post = {
        "schema": "fixture_current_post",
        "source_count": 15,
        "source_sha256": _source_map(
            root,
            prefix="current_post",
            count=15,
            mainline=False,
        ),
        "official_test_accessed": False,
        "training_source_lock": {
            "path": str(lock_by_name["current_training_48"].path),
            "sha256": current_training_sha,
        },
    }
    _write_json(
        lock_by_name["current_posttraining_15"].path,
        current_post,
    )
    current_post_sha = _sha(
        lock_by_name["current_posttraining_15"].path
    )

    paired_parent = dict(parent)
    paired_parent.pop("variant")
    paired_training = {
        "schema": "fixture_paired_training",
        "source_count": 51,
        "source_sha256": _source_map(
            root,
            prefix="paired_training",
            count=51,
            mainline=True,
        ),
        "formal_contract": {"official_test_accessed": False},
        "parent_checkpoint": paired_parent,
        "upstream_source_lock_sha256": current_training_sha,
    }
    _write_json(
        lock_by_name["paired_training_51"].path,
        paired_training,
    )
    paired_training_sha = _sha(
        lock_by_name["paired_training_51"].path
    )

    paired_post = {
        "schema": "fixture_paired_post_v2",
        "source_count": 12,
        "source_sha256": _source_map(
            root,
            prefix="paired_post",
            count=12,
            mainline=False,
        ),
        "official_test_accessed": False,
        "training_source_lock": {
            "path": str(lock_by_name["paired_training_51"].path),
            "sha256": paired_training_sha,
        },
        "reference_closure_source_lock": {
            "path": str(lock_by_name["current_posttraining_15"].path),
            "sha256": current_post_sha,
        },
    }
    _write_json(
        lock_by_name["paired_posttraining_v2_12"].path,
        paired_post,
    )
    return {
        "current_training_48": current_training_sha,
        "current_posttraining_15": current_post_sha,
        "paired_training_51": paired_training_sha,
        "paired_posttraining_v2_12": _sha(
            lock_by_name["paired_posttraining_v2_12"].path
        ),
    }


def _make_commands(layout: manifest.EvidenceLayout) -> None:
    for command in layout.commands:
        _write(
            command.source,
            f"fixture command {command.name}\n".encode(),
        )
        command.source.chmod(0o755)


def _make_template(layout: manifest.EvidenceLayout) -> None:
    source = (
        manifest.REPO_ROOT
        / "experiments/"
        "tpd_ner_v4_qfg_v2_croa_reproducibility_manifest_template.md"
    )
    _write(layout.template_path, source.read_bytes())


def _make_run(
    spec: manifest.RunSpec,
    *,
    training_lock_sha256: str,
    parent: Path,
) -> list[str]:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_entries: dict[str, dict[str, object]] = {}
    checkpoint_sha: dict[str, str] = {}
    epochs = {"best.pth.tar": 17, "best_miou.pth.tar": 23}
    for filename, (role, _) in manifest.CHECKPOINTS.items():
        path = spec.run_dir / filename
        _write(path, f"{spec.method_id}:{filename}".encode())
        digest = _sha(path)
        checkpoint_sha[filename] = digest
        key = "best" if filename == "best.pth.tar" else "best_miou"
        checkpoint_entries[key] = {
            "path": str(path),
            "sha256": digest,
            "epoch": epochs[filename],
            "role": role,
        }
    identity = {
        "schema": "fixture_run_identity",
        "run_id": f"fixture:{spec.method_id}",
        "dataset": manifest.DATASET,
        "seed": manifest.TRAINING_SEED,
        "split_seed": manifest.SPLIT_SEED,
        "source_locks": {
            spec.source_lock_key: training_lock_sha256,
            "parent_checkpoint": _sha(parent),
        },
        "training_contract": {
            "environment": {
                "physical_gpu_index": spec.gpu_index,
                "physical_gpu_uuid": spec.gpu_uuid,
                "visible_cuda_device_count": 1,
            }
        },
    }
    protocol = {
        "schema": "fixture_protocol",
        "official_test_accessed": False,
        "arguments": {
            "dataset": manifest.DATASET,
            "seed": manifest.TRAINING_SEED,
            "split_seed": manifest.SPLIT_SEED,
            "epochs": manifest.FORMAL_EPOCHS,
            "variant": spec.variant,
            "parent_checkpoint": str(parent),
        },
        "parent_checkpoint": {
            "path": str(parent),
            "sha256": _sha(parent),
            "epoch": 489,
        },
        "run_identity": identity,
    }
    summary = {
        "schema": "fixture_summary",
        "status": "complete",
        "variant": spec.variant,
        "dataset": manifest.DATASET,
        "seed": manifest.TRAINING_SEED,
        "split_seed": manifest.SPLIT_SEED,
        "official_test_accessed": False,
        "formal_contract": {
            "epochs": manifest.FORMAL_EPOCHS,
            "official_test_accessed": False,
        },
        "run_identity": identity,
        "best_epoch": epochs["best.pth.tar"],
        "best_miou_epoch": epochs["best_miou.pth.tar"],
        "best_checkpoint": str(spec.run_dir / "best.pth.tar"),
        "best_miou_checkpoint": str(
            spec.run_dir / "best_miou.pth.tar"
        ),
        "checkpoints": checkpoint_entries,
    }
    split = {
        "dataset": manifest.DATASET,
        "split_seed": manifest.SPLIT_SEED,
        "used_train_count": 530,
        "used_val_count": 133,
        "official_test_accessed": False,
    }
    _write_json(spec.run_dir / "protocol.json", protocol)
    _write_json(spec.run_dir / "summary.json", summary)
    _write_json(spec.run_dir / "split.json", split)
    _write(
        spec.run_dir / "metrics.jsonl",
        "".join(
            json.dumps({"epoch": epoch}, sort_keys=True) + "\n"
            for epoch in range(1, manifest.FORMAL_EPOCHS + 1)
        ).encode(),
    )
    sweep_shas: list[str] = []
    for filename, (role, _) in manifest.CHECKPOINTS.items():
        sweep = {
            "schema": "fixture_sweep",
            "dataset": manifest.DATASET,
            "seed": manifest.TRAINING_SEED,
            "split_seed": manifest.SPLIT_SEED,
            "variant": spec.variant,
            "run_directory": str(spec.run_dir),
            "checkpoint": str(spec.run_dir / filename),
            "checkpoint_role": role,
            "checkpoint_epoch": epochs[filename],
            "checkpoint_sha256": checkpoint_sha[filename],
            "validation_split_sha256": (
                manifest.VALIDATION_SPLIT_SHA256
            ),
            "fixed_threshold_0_5": {"threshold": 0.5},
            "best_points_under_fa_budget": {
                key: {"threshold": 0.5}
                for key in manifest.BUDGET_KEYS
            },
            "points": [{"threshold": 0.5}],
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
            "official_test_accessed": False,
        }
        sweep_path = spec.run_dir / manifest.SWEEP_NAMES[filename]
        _write_json(sweep_path, sweep)
        sweep_shas.append(_sha(sweep_path))
    return sweep_shas


def _selected(
    spec: manifest.RunSpec,
) -> dict[str, object]:
    checkpoint = spec.run_dir / "best.pth.tar"
    return {
        "method_id": spec.method_id,
        "variant": spec.variant,
        "checkpoint": checkpoint.name,
        "checkpoint_role": "best_validation_pd_primary",
        "role_name": "pd_primary",
        "checkpoint_epoch": 17,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha(checkpoint),
        "threshold": 0.5,
        "metrics": {
            "pd": 188 / 189,
            "fa": 1e-6,
            "miou": 0.93,
            "tiny_pd": 1.0,
            "false_objects_per_image": 0.01,
        },
        "checkpoint_local_atomic_point": True,
    }


def _make_closure(
    spec: manifest.ClosureSpec,
    *,
    selected_run: manifest.RunSpec,
    sweep_shas: list[str],
    closure_lock_path: Path,
    current_reference_shas: list[str] | None = None,
) -> None:
    selected = _selected(selected_run)
    all_input_shas = [
        *sweep_shas,
        *(current_reference_shas or []),
    ]
    if spec.factorial_json is not None:
        factorial = {
            "schema": "fixture_factorial",
            "status": "complete",
            "dataset": manifest.DATASET,
            "training_seed": manifest.TRAINING_SEED,
            "split_seed": manifest.SPLIT_SEED,
            "official_test_accessed": False,
            "artifact_bindings": {
                f"binding_{index}": {"sha256": digest}
                for index, digest in enumerate(sweep_shas)
            },
        }
        _write_json(spec.factorial_json, factorial)
        _write(spec.factorial_markdown, b"# fixture factorial\n")
    selection = {
        "schema": "fixture_selection",
        "status": "complete",
        "dataset": manifest.DATASET,
        "training_seed": manifest.TRAINING_SEED,
        "split_seed": manifest.SPLIT_SEED,
        "official_test_accessed": False,
        "selected_method_id": selected["method_id"],
        "input_bindings": {
            f"binding_{index}": {"sha256": digest}
            for index, digest in enumerate(all_input_shas)
        },
        "deployment_selection": {
            "selected": selected,
            "cross_checkpoint_metric_stitching": False,
        },
    }
    _write_json(spec.selection_json, selection)
    _write(spec.selection_markdown, b"# fixture selection\n")
    _write(
        spec.deployment_artifact,
        f"fixture export {spec.name}".encode(),
    )
    deployment = {
        "schema": "fixture_deployment",
        "status": "complete",
        "dataset": manifest.DATASET,
        "training_seed": manifest.TRAINING_SEED,
        "split_seed": manifest.SPLIT_SEED,
        "official_test_accessed": False,
        "selected_method_id": selected["method_id"],
        "selected_variant": selected["variant"],
        "selected_checkpoint": {
            key: selected[key]
            for key in (
                "checkpoint",
                "checkpoint_role",
                "role_name",
                "checkpoint_epoch",
                "checkpoint_path",
                "checkpoint_sha256",
            )
        },
        "deployment_operating_point": {"selected": selected},
        "final_selection": {
            "path": str(spec.selection_json),
            "sha256": _sha(spec.selection_json),
        },
        "posttraining_closure_source_lock": {
            "path": str(closure_lock_path),
            "sha256": _sha(closure_lock_path),
        },
        "artifact": {
            "path": str(spec.deployment_artifact),
            "sha256": _sha(spec.deployment_artifact),
        },
        "export_mode": "fixture_head_free_export",
    }
    _write_json(spec.deployment_manifest, deployment)


def _make_receipt(
    layout: manifest.EvidenceLayout,
    *,
    family: str,
    status: str = "complete",
) -> None:
    current = layout.current_closure
    current_lock = {
        lock.name: lock.path for lock in layout.locks
    }["current_posttraining_15"]
    paired_command = {
        command.name: command for command in layout.commands
    }["paired_training_2x5090"]
    paired_lock = {
        lock.name: lock.path for lock in layout.locks
    }["paired_training_51"]
    current_selection = json.loads(
        current.selection_json.read_text(encoding="utf-8")
    )
    paired = family == "paired"
    receipt = {
        "schema": manifest.CONTROLLER_SCHEMA,
        "status": status,
        "phase": (
            "paired_training_complete"
            if paired and status == "complete"
            else (
                "no_fallback"
                if status == "complete"
                else "paired_launching"
            )
        ),
        "authoritative_action": "launch_paired" if paired else "no_fallback",
        "paired_required": paired,
        "terminal_for_fallback_controller": status == "complete",
        "terminal_for_reproducibility_manifest": (
            status == "complete" and not paired
        ),
        "receipt_write_policy": "atomic_state_transition",
        "paired_training_complete": paired and status == "complete",
        "official_test_accessed": False,
        "selected_method_id": current_selection["selected_method_id"],
        "selected_candidate_status": "RELATIVE_IMPROVED",
        "query_fg_stage_success": not paired,
        "meaningful_overall_improvement_by_frozen_policy": True,
        "current_terminal": {
            "final_selection": {
                "path": str(current.selection_json),
                "sha256": _sha(current.selection_json),
            },
            "deployment_manifest": {
                "path": str(current.deployment_manifest),
                "sha256": _sha(current.deployment_manifest),
            },
            "deployment_artifact": {
                "path": str(current.deployment_artifact),
                "sha256": _sha(current.deployment_artifact),
            },
            "posttraining_source_lock": {
                "path": str(current_lock),
                "sha256": _sha(current_lock),
            },
        },
        "launcher": {
            "path": str(paired_command.source),
            "sha256": _sha(paired_command.source),
            "verified_regular_executable": True,
            "fixed_physical_gpus": [2, 3],
            "wait_for_gpu_idle": False,
            "paired_flock": True,
            "result_root": str(
                layout.paired_runs[0].run_dir.parents[4]
            ),
            "source_lock": {
                "path": str(paired_lock),
                "sha256": _sha(paired_lock),
                "schema": "fixture_paired_training",
                "source_count": 51,
            },
            "invoked": paired,
            "exit_status": 0 if paired else None,
        },
        "paired": {
            "required": paired,
            "training_complete": paired and status == "complete",
            "posttraining_selection_complete": False,
            "posttraining_deployment_complete": False,
            "result_root": str(
                layout.paired_runs[0].run_dir.parents[4]
            ),
            **(
                {
                    "formal_epochs": manifest.FORMAL_EPOCHS,
                    "training_seed": manifest.TRAINING_SEED,
                    "split_seed": manifest.SPLIT_SEED,
                    "source_lock": {
                        "path": str(paired_lock),
                        "sha256": _sha(paired_lock),
                        "schema": "fixture_paired_training",
                        "source_count": 51,
                    },
                    "runs": {
                        run.method_id: {
                            "method_id": run.method_id,
                            "formal_epochs_complete": (
                                manifest.FORMAL_EPOCHS
                            ),
                            "files": {
                                "summary": {
                                    "path": str(
                                        run.run_dir / "summary.json"
                                    ),
                                    "sha256": _sha(
                                        run.run_dir / "summary.json"
                                    ),
                                },
                                "metrics": {
                                    "path": str(
                                        run.run_dir / "metrics.jsonl"
                                    ),
                                    "sha256": _sha(
                                        run.run_dir / "metrics.jsonl"
                                    ),
                                },
                                "pd_primary_checkpoint": {
                                    "path": str(
                                        run.run_dir / "best.pth.tar"
                                    ),
                                    "sha256": _sha(
                                        run.run_dir / "best.pth.tar"
                                    ),
                                },
                                "miou_secondary_checkpoint": {
                                    "path": str(
                                        run.run_dir
                                        / "best_miou.pth.tar"
                                    ),
                                    "sha256": _sha(
                                        run.run_dir
                                        / "best_miou.pth.tar"
                                    ),
                                },
                            },
                        }
                        for run in layout.paired_runs
                    },
                }
                if paired
                else {"runs": {}}
            ),
        },
    }
    _write_json(layout.controller_receipt, receipt)


def _fixture(
    tmp_path: Path,
    *,
    family: str,
    make_paired_closure: bool = True,
    receipt_status: str = "complete",
) -> manifest.EvidenceLayout:
    layout = manifest.default_layout(
        repo_root=tmp_path,
        output_dir=tmp_path / "sealed_manifest",
        enforce_frozen_lock_digests=False,
    )
    _make_template(layout)
    _make_commands(layout)
    lock_shas = _make_locks(layout)
    current_sweeps: list[str] = []
    for spec in layout.current_runs:
        current_sweeps.extend(
            _make_run(
                spec,
                training_lock_sha256=lock_shas["current_training_48"],
                parent=layout.parent_checkpoint,
            )
        )
    lock_paths = {lock.name: lock.path for lock in layout.locks}
    _make_closure(
        layout.current_closure,
        selected_run=layout.current_runs[0],
        sweep_shas=current_sweeps,
        closure_lock_path=lock_paths["current_posttraining_15"],
    )
    if family == "paired":
        paired_sweeps: list[str] = []
        for spec in layout.paired_runs:
            paired_sweeps.extend(
                _make_run(
                    spec,
                    training_lock_sha256=lock_shas["paired_training_51"],
                    parent=layout.parent_checkpoint,
                )
            )
        if make_paired_closure:
            _make_closure(
                layout.paired_closure,
                selected_run=layout.paired_runs[0],
                sweep_shas=paired_sweeps,
                closure_lock_path=lock_paths[
                    "paired_posttraining_v2_12"
                ],
                current_reference_shas=[
                    lock_shas["current_posttraining_15"],
                    *current_sweeps,
                ],
            )
    _make_receipt(layout, family=family, status=receipt_status)
    return layout


def test_in_progress_receipt_returns_75_without_partial_output(
    tmp_path: Path,
) -> None:
    layout = _fixture(
        tmp_path,
        family="current",
        receipt_status="in_progress",
    )
    with mock.patch.object(
        manifest,
        "default_layout",
        return_value=layout,
    ):
        assert manifest.main(["--terminal-family", "current"]) == 75
    assert not layout.output_dir.exists()


def test_current_bundle_is_atomic_write_once_and_idempotent(
    tmp_path: Path,
) -> None:
    layout = _fixture(tmp_path, family="current")
    first = manifest.execute(layout, terminal_family="current")
    second = manifest.execute(layout, terminal_family="current")
    assert first["action"] == "publish"
    assert first["atomic_pair"] is True
    assert second["action"] == "verify"
    assert second["writes_performed"] is False
    assert {path.name for path in layout.output_dir.iterdir()} == {
        "manifest.json",
        "manifest.md",
    }
    payload = json.loads(
        (layout.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert payload["terminal_family"] == "current"
    assert payload["terminal_authority"][
        "own_best_best_miou_sweep_count"
    ] == 4
    assert payload["claim_boundary"]["official_test_accessed"] is False


def test_existing_conflict_is_refused_without_overwrite(
    tmp_path: Path,
) -> None:
    layout = _fixture(tmp_path, family="current")
    manifest.execute(layout, terminal_family="current")
    json_before = (layout.output_dir / "manifest.json").read_bytes()
    (layout.output_dir / "manifest.md").write_text(
        "conflict\n",
        encoding="utf-8",
    )
    with pytest.raises(manifest.EvidenceConflict):
        manifest.execute(layout, terminal_family="current")
    assert (layout.output_dir / "manifest.json").read_bytes() == json_before
    assert (
        layout.output_dir / "manifest.md"
    ).read_text(encoding="utf-8") == "conflict\n"


def test_paired_receipt_alone_cannot_seal_without_posttraining_closure(
    tmp_path: Path,
) -> None:
    layout = _fixture(
        tmp_path,
        family="paired",
        make_paired_closure=False,
    )
    with pytest.raises(manifest.IncompleteEvidence):
        manifest.execute(layout, terminal_family="paired")
    assert not layout.output_dir.exists()


def test_paired_terminal_collects_ef_own_checkpoints_and_four_sweeps(
    tmp_path: Path,
) -> None:
    layout = _fixture(tmp_path, family="paired")
    action = manifest.execute(layout, terminal_family="paired")
    assert action["action"] == "publish"
    payload = json.loads(
        (layout.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(payload["runs"]["paired"]) == {
        "e_qfg_dlr",
        "f_tss_qfg_dlr",
    }
    assert payload["terminal_authority"][
        "own_best_best_miou_sweep_count"
    ] == 4
    for run in payload["runs"]["paired"].values():
        assert set(run["checkpoints"]) == {"best", "best_miou"}
        assert run["own_best_and_best_miou"] is True


def test_any_official_test_true_claim_is_rejected(
    tmp_path: Path,
) -> None:
    layout = _fixture(tmp_path, family="current")
    sweep = (
        layout.current_runs[0].run_dir
        / manifest.SWEEP_NAMES["best.pth.tar"]
    )
    payload = json.loads(sweep.read_text(encoding="utf-8"))
    payload["official_test_accessed"] = True
    _write_json(sweep, payload)
    with pytest.raises(manifest.EvidenceConflict):
        manifest.execute(layout, terminal_family="current")
    assert not layout.output_dir.exists()
