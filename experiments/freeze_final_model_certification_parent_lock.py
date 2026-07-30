#!/usr/bin/env python3
"""Plan, write once, or verify the final-model certification parent lock.

The parent lock binds the already frozen seed-42 model, data, evaluator, and
deployment facts used as the upstream authority for later certification work.
It deliberately excludes itself, this freezer, the certification protocol,
and any future certification commit.  A later release attestation binds the
certification commit without creating a commit/self-reference cycle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_ner_v4_qfg_v2_croa_operational_closure_v2
    as operational_freezer,
)


LOCK_SCHEMA = "sctransnet_final_model_certification_parent_lock_v1"
PLAN_SCHEMA = "sctransnet_final_model_certification_parent_lock_plan_v1"
ACTION_SCHEMA = "sctransnet_final_model_certification_parent_lock_action_v1"
LOCK_KIND = "certification_parent"

FROZEN_MODEL_SOURCE_COMMIT = (
    "a295f751470c3414bb453d702451cecde41a1524"
)
FROZEN_MODEL_SOURCE_TREE = (
    "e12457685d6eb61252adc757df37e42c04a3cca6"
)
FROZEN_MODEL_SOURCE_PARENT = (
    "5bc3ea5cbf4bcbc12c285d0d85fdfb74ab9cc7dc"
)
FROZEN_MODEL_SOURCE_TIMESTAMP = 1785369456
FROZEN_MODEL_SOURCE_SUBJECT = (
    "Add QFG V2 CROA experiment and deployment workflow"
)

DEFAULT_OUTPUT_RELATIVE_PATH = (
    "experiments/final_model_certification_parent_lock_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT_RELATIVE_PATH

DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
TRAIN_COUNT = 530
VALIDATION_COUNT = 133
TRAIN_IDS_SHA256 = (
    "9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b"
)
VALIDATION_IDS_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
NORMALIZATION_MEAN = 107.28969064388635
NORMALIZATION_STD = 32.74261895755552

PARENT_CHECKPOINT_SHA256 = (
    "0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab"
)
PARENT_STATE_DICT_SHA256 = (
    "2b8249ffd86866597f376c80839395a3cbdbb72a68301cd8a5a6eb36595c7e75"
)
D_CHECKPOINT_SHA256 = (
    "890c8cf0e0f7c3a4c21e5772e69cd89e3038b308a1d77be58365f2254b89b678"
)
INFERENCE_ARTIFACT_SHA256 = (
    "997027bb2cc59e0e16ef85beba2c78ab8b3e195de962acbe7c97adc8c007c63a"
)

DEFAULT_THRESHOLD = 0.5
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
MATCH_RADIUS = 3.0
TINY_AREA = 9

RESULT_ROOT = (
    "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/NUDT-SIRST"
)
PARENT_CHECKPOINT_PATH = (
    "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/"
    "seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar"
)
D_RUN_ROOT = (
    f"{RESULT_ROOT}/tss_qfg/seed_42_formal800_tss_qfg"
)
D_CHECKPOINT_PATH = f"{D_RUN_ROOT}/best_miou.pth.tar"
INFERENCE_ARTIFACT_PATH = (
    f"{RESULT_ROOT}/deployment/"
    "tpd_ner_v4_qfg_v2_croa_formal800_inference.pth.tar"
)


@dataclass(frozen=True)
class ArtifactContract:
    role: str
    relative_path: str
    sha256: str
    schema: str | None = None


ARTIFACT_CONTRACTS = (
    ArtifactContract(
        "initialization_parent_checkpoint",
        PARENT_CHECKPOINT_PATH,
        PARENT_CHECKPOINT_SHA256,
    ),
    ArtifactContract(
        "d_training_checkpoint",
        D_CHECKPOINT_PATH,
        D_CHECKPOINT_SHA256,
    ),
    ArtifactContract(
        "final_inference_artifact",
        INFERENCE_ARTIFACT_PATH,
        INFERENCE_ARTIFACT_SHA256,
    ),
    ArtifactContract(
        "deployment_manifest_v2",
        (
            f"{RESULT_ROOT}/deployment_v2/"
            "tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest_v2.json"
        ),
        "542f8df7fe11cf982835bccba4bf7ae12c2d3538de16fa14b012b99732bc0faf",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_manifest_v2",
    ),
    ArtifactContract(
        "default_operating_point_v2",
        (
            f"{RESULT_ROOT}/deployment_v2/"
            "tpd_ner_v4_qfg_v2_croa_formal800_default_operating_point_v2.json"
        ),
        "49bce98376004ff611723ab03aa608b8ba2192f85ba408c09d8bfb4580735a0b",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2",
    ),
    ArtifactContract(
        "final_selection_v2",
        (
            f"{RESULT_ROOT}/final_selection/"
            "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json"
        ),
        "29d4b3e21c7308c97e2abc2ae28016ff71ece71d88b758b0e3ae2ce7c92988a8",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_final_selection_v2",
    ),
    ArtifactContract(
        "d_run_split",
        f"{D_RUN_ROOT}/split.json",
        "391bca28848038d6386a6c70cbaeb902ba71a8dc73a4a134441ac2aa5b438828",
    ),
    ArtifactContract(
        "d_run_protocol",
        f"{D_RUN_ROOT}/protocol.json",
        "83ee7e9d937b55dfa46344bfbe2fea44feef26629b669f85eca0bc49e8ce00ae",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_entry_v1",
    ),
    ArtifactContract(
        "d_selected_checkpoint_sweep",
        f"{D_RUN_ROOT}/pd_fa_sweep_best_miou.pth.json",
        "37cb99e061abc49e2836c81eab717efd6dcb840cc7ffde2fe3ded9e6a7b659f5",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_checkpoint_local_pd_fa_v1",
    ),
    ArtifactContract(
        "seed42_factorial",
        (
            f"{RESULT_ROOT}/comparison_factorial_v1/"
            "tss_qfg_v2_croa_factorial_seed42.json"
        ),
        "aa4f2867e6b9c49b66d63b2b9f5210c0ac89d1e19595b23ac3c0b347fc05344e",
        "sctransnet_tss_qfg_v2_croa_factorial_seed42_v2",
    ),
    ArtifactContract(
        "reproducibility_manifest_v2",
        f"{RESULT_ROOT}/reproducibility_manifest_v2/manifest.json",
        "410f87ec069bbd32ae71b80e2655608867665a15c91e5493d2d02e2badef9a38",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_formal800_reproducibility_manifest_v2",
    ),
    ArtifactContract(
        "training_source_lock",
        "experiments/tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json",
        "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_source_lock_v1",
    ),
    ArtifactContract(
        "posttraining_source_lock",
        "experiments/tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json",
        "315f091b75078e65b871946cecae92893e8915bb3951b6fc4dcf3a52c984cbbd",
        "sctransnet_tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock_v1",
    ),
    ArtifactContract(
        "operational_source_lock_v2",
        (
            "experiments/"
            "tpd_ner_v4_qfg_v2_croa_operational_closure_source_lock_v2.json"
        ),
        "ea23c5a13f0a85841e624ecb57863a6f36ef711ff5542de4aa945d09cb76b182",
        (
            "sctransnet_tpd_ner_v4_qfg_v2_croa_"
            "operational_closure_source_lock_v2"
        ),
    ),
)

FROZEN_SOURCE_SHA256 = {
    "dataset.py": (
        "516ea9c410f80cc9ae912cf0443126a067dd14b6cc5ad7945e83cfc497f4678d"
    ),
    "experiments/evaluate_pd_fa_sweep.py": (
        "0224ab44dc346ebdbd4cb4775c493bd6eecdc877019832dea0f16e59ab353537"
    ),
    "experiments/evaluate_tpd_clean_v6_pd_fa.py": (
        "bc1f8a25b0f047719002ed838442fe0e244d25292e2bd67f768f9b08d7251a5f"
    ),
    "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py": (
        "687617a96aeec7861b99d760f520b801abd46e3c820a12cc967e326130cf450a"
    ),
    (
        "experiments/"
        "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py"
    ): (
        "3d1559101f498a49d9a2afdb8bc49c8110ab36a03e33c5da9502a6d44086fa61"
    ),
    (
        "experiments/"
        "publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2.py"
    ): (
        "96ba295cb6106a09a870337a7d01fde9d694d867ec14a89aa1692fe31f18fe4b"
    ),
    "experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py": (
        "fe1e01ca901584a9558287b332bedcf87157d194f4ea51a61559fed3ea8853de"
    ),
    (
        "model/"
        "tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py"
    ): (
        "c8c7db1fc8b3e83c45ee11dcb45f7c09dd2f4456554c4267aaba3b369394ff53"
    ),
}

SOURCE_LOCK_ROLES = {
    "training_source_lock": 48,
    "posttraining_source_lock": 15,
    "operational_source_lock_v2": 4,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ValueError(f"SHA-256 input must be a regular file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_repo(repo_root: Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {root}")
    return root


def _repo_file(
    repo_root: Path,
    relative_path: str,
    label: str,
) -> Path:
    pure = PurePosixPath(relative_path)
    _require(
        relative_path == pure.as_posix()
        and not pure.is_absolute()
        and ".." not in pure.parts
        and "." not in pure.parts,
        f"{label} is not a canonical repository-relative path: {relative_path}",
    )
    raw = repo_root.joinpath(*pure.parts)
    if raw.is_symlink() or not raw.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {raw}")
    resolved = raw.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{label} lies outside the repository: {raw}") from exc
    _require(
        resolved == raw,
        f"{label} contains a non-canonical or linked path component: {raw}",
    )
    return raw


def _load_json_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return raw, value


def _git(
    repo_root: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    command = ["git", "-C", str(repo_root), *arguments]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=text,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.strip()
            if isinstance(completed.stderr, str)
            else completed.stderr.decode("utf-8", errors="replace").strip()
        )
        raise ValueError(
            f"Git command failed ({' '.join(arguments)}): {stderr}"
        )
    return completed.stdout


def _commit_binding(repo_root: Path) -> dict[str, Any]:
    commit_type = str(
        _git(repo_root, "cat-file", "-t", FROZEN_MODEL_SOURCE_COMMIT)
    ).strip()
    _require_equal("frozen object type", commit_type, "commit")
    tree = str(
        _git(
            repo_root,
            "show",
            "-s",
            "--format=%T",
            FROZEN_MODEL_SOURCE_COMMIT,
        )
    ).strip()
    parents = str(
        _git(
            repo_root,
            "show",
            "-s",
            "--format=%P",
            FROZEN_MODEL_SOURCE_COMMIT,
        )
    ).strip()
    timestamp = int(
        str(
            _git(
                repo_root,
                "show",
                "-s",
                "--format=%ct",
                FROZEN_MODEL_SOURCE_COMMIT,
            )
        ).strip()
    )
    subject = str(
        _git(
            repo_root,
            "show",
            "-s",
            "--format=%s",
            FROZEN_MODEL_SOURCE_COMMIT,
        )
    ).strip()
    _require_equal("frozen commit tree", tree, FROZEN_MODEL_SOURCE_TREE)
    _require_equal(
        "frozen commit parents",
        parents,
        FROZEN_MODEL_SOURCE_PARENT,
    )
    _require_equal(
        "frozen commit timestamp",
        timestamp,
        FROZEN_MODEL_SOURCE_TIMESTAMP,
    )
    _require_equal(
        "frozen commit subject",
        subject,
        FROZEN_MODEL_SOURCE_SUBJECT,
    )
    return {
        "commit": FROZEN_MODEL_SOURCE_COMMIT,
        "tree": tree,
        "parents": [FROZEN_MODEL_SOURCE_PARENT],
        "commit_timestamp": timestamp,
        "subject": subject,
    }


def _frozen_source_bindings(
    repo_root: Path,
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for relative, expected_sha256 in sorted(FROZEN_SOURCE_SHA256.items()):
        live = _repo_file(repo_root, relative, f"frozen source {relative}")
        live_sha256 = sha256_file(live)
        _require_equal(
            f"live frozen source SHA {relative}",
            live_sha256,
            expected_sha256,
        )
        object_name = f"{FROZEN_MODEL_SOURCE_COMMIT}:{relative}"
        commit_bytes = _git(
            repo_root,
            "cat-file",
            "blob",
            object_name,
            text=False,
        )
        _require(isinstance(commit_bytes, bytes), "Git blob output is not bytes")
        commit_sha256 = hashlib.sha256(commit_bytes).hexdigest()
        _require_equal(
            f"commit/live source SHA {relative}",
            commit_sha256,
            live_sha256,
        )
        blob_oid = str(
            _git(repo_root, "rev-parse", object_name)
        ).strip()
        bindings[relative] = {
            "sha256": live_sha256,
            "git_blob_oid": blob_oid,
        }
    return bindings


def _assert_official_test_false(value: Any, label: str) -> None:
    false_keys = {
        "official_test_accessed",
        "official_test_claim",
        "cross_seed_stability_claim",
    }

    def visit(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                next_location = f"{location}.{key}"
                if key in false_keys:
                    _require_equal(next_location, child, False)
                visit(child, next_location)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")

    visit(value, label)


def _validate_artifacts(
    repo_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    roles = [contract.role for contract in ARTIFACT_CONTRACTS]
    _require_equal(
        "artifact role uniqueness",
        len(set(roles)),
        len(roles),
    )
    paths = [contract.relative_path for contract in ARTIFACT_CONTRACTS]
    _require_equal(
        "artifact path uniqueness",
        len(set(paths)),
        len(paths),
    )

    records: dict[str, dict[str, Any]] = {}
    json_payloads: dict[str, dict[str, Any]] = {}
    for contract in ARTIFACT_CONTRACTS:
        path = _repo_file(
            repo_root,
            contract.relative_path,
            contract.role,
        )
        digest = sha256_file(path)
        _require_equal(
            f"{contract.role} SHA-256",
            digest,
            contract.sha256,
        )
        record: dict[str, Any] = {
            "path": contract.relative_path,
            "sha256": digest,
        }
        if contract.schema is not None:
            _, payload = _load_json_file(path, contract.role)
            _require_equal(
                f"{contract.role} schema",
                payload.get("schema"),
                contract.schema,
            )
            record["schema"] = contract.schema
            json_payloads[contract.role] = payload
        elif contract.relative_path.endswith(".json"):
            _, payload = _load_json_file(path, contract.role)
            json_payloads[contract.role] = payload
        records[contract.role] = record
    return records, json_payloads


def _validate_source_lock(
    repo_root: Path,
    role: str,
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_count: int,
) -> None:
    lock_path = _repo_file(repo_root, str(record["path"]), role)
    raw = lock_path.read_bytes()
    _require_equal(
        f"{role} canonical JSON",
        raw,
        canonical_json_bytes(payload),
    )
    _require_equal(
        f"{role} source_count",
        payload.get("source_count"),
        expected_count,
    )
    source_sha256 = payload.get("source_sha256")
    _require(
        isinstance(source_sha256, Mapping),
        f"{role} source_sha256 is missing",
    )
    _require_equal(
        f"{role} source hash count",
        len(source_sha256),
        expected_count,
    )
    for relative, expected in source_sha256.items():
        _require(
            isinstance(relative, str) and _is_sha256(expected),
            f"{role} contains an invalid source binding: {relative!r}",
        )
        source = _repo_file(
            repo_root,
            relative,
            f"{role} live source {relative}",
        )
        _require_equal(
            f"{role} live source SHA {relative}",
            sha256_file(source),
            expected,
        )
    _assert_official_test_false(payload, role)


def _validate_source_locks(
    repo_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    for role, count in SOURCE_LOCK_ROLES.items():
        _validate_source_lock(
            repo_root,
            role,
            payloads[role],
            records[role],
            count,
        )
    operational_result = operational_freezer.verify(
        repo_root,
        repo_root / records["operational_source_lock_v2"]["path"],
    )
    _require_equal(
        "reused operational source-lock verifier",
        operational_result.get("verified"),
        True,
    )


def _validate_semantic_contracts(
    records: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    final = payloads["final_selection_v2"]
    for field, expected in (
        ("status", "complete"),
        ("decision", "SELECT_D_TSS_QFG"),
        ("scope", "single_seed_internal_validation"),
        ("training_seed", TRAINING_SEED),
        ("split_seed", SPLIT_SEED),
        ("validation_count", VALIDATION_COUNT),
        ("validation_split_sha256", VALIDATION_IDS_SHA256),
        ("final_model_established", True),
        ("final_model_engineering_selected", True),
        ("paper_core_established", False),
        ("stability_claim_supported", False),
        ("official_test_accessed", False),
    ):
        _require_equal(f"final selection {field}", final.get(field), expected)
    selection = final.get("selection")
    _require(isinstance(selection, Mapping), "final selection block is missing")
    _require_equal(
        "final selected method",
        selection.get("selected_method_id"),
        "d_tss_qfg",
    )
    _require_equal(
        "final selected variant",
        selection.get("selected_variant"),
        "tss_qfg",
    )
    _require_equal(
        "final training uses TSS",
        selection.get("final_training_uses_tss"),
        True,
    )
    _require_equal(
        "final inference uses TSS",
        selection.get("final_inference_uses_tss"),
        False,
    )

    deployment = payloads["deployment_manifest_v2"]
    for field, expected in (
        ("status", "complete"),
        ("dataset", DATASET),
        ("selected_method_id", "d_tss_qfg"),
        ("selected_variant", "tss_qfg"),
        ("training_seed", TRAINING_SEED),
        ("split_seed", SPLIT_SEED),
        ("official_test_accessed", False),
        ("checkpoint_unchanged", True),
        ("weights_unchanged", True),
        ("selected_point_is_checkpoint_local", True),
    ):
        _require_equal(f"deployment {field}", deployment.get(field), expected)
    selected_checkpoint = deployment.get("selected_checkpoint")
    _require(
        isinstance(selected_checkpoint, Mapping),
        "deployment selected checkpoint is missing",
    )
    for field, expected in (
        ("checkpoint", "best_miou.pth.tar"),
        ("checkpoint_epoch", 3),
        ("checkpoint_role", "best_validation_miou_secondary"),
        ("checkpoint_sha256", D_CHECKPOINT_SHA256),
    ):
        _require_equal(
            f"deployment checkpoint {field}",
            selected_checkpoint.get(field),
            expected,
        )
    artifact = deployment.get("artifact")
    _require(isinstance(artifact, Mapping), "deployment artifact is missing")
    for field, expected in (
        ("sha256", INFERENCE_ARTIFACT_SHA256),
        ("source_checkpoint_sha256", D_CHECKPOINT_SHA256),
        ("strict_load", True),
        ("survival_state_absent", True),
        ("qfg_state_preserved", True),
        ("inference_parameter_count", 10870130),
        ("inference_state_key_count", 564),
    ):
        _require_equal(
            f"deployment artifact {field}",
            artifact.get(field),
            expected,
        )
    deployment_point = deployment.get("deployment_operating_point", {})
    deployment_selected = deployment_point.get("selected", {})
    _require_equal(
        "deployment operating-point threshold",
        deployment_selected.get("threshold"),
        DEFAULT_THRESHOLD,
    )
    _require_equal(
        "deployment operating-point source",
        deployment_selected.get("source"),
        "fixed_threshold_0_5",
    )
    _require_equal(
        "deployment operating-point method",
        deployment_selected.get("method_id"),
        "d_tss_qfg",
    )

    profile = payloads["default_operating_point_v2"]
    for field, expected in (
        ("status", "complete"),
        ("selected_method_id", "d_tss_qfg"),
        ("selected_variant", "tss_qfg"),
        ("training_seed", TRAINING_SEED),
        ("split_seed", SPLIT_SEED),
        ("official_test_accessed", False),
    ):
        _require_equal(f"default profile {field}", profile.get(field), expected)
    profile_point = profile.get("default_operating_point", {})
    _require_equal(
        "default profile threshold",
        profile_point.get("threshold"),
        DEFAULT_THRESHOLD,
    )
    _require_equal(
        "default profile source",
        profile_point.get("source"),
        "fixed_threshold_0_5",
    )
    metrics = profile_point.get("metrics", {})
    for field, expected in (
        ("matched_target_count", 188),
        ("target_count", 189),
        ("fa", 4.1301985432330825e-6),
        ("miou", 0.9370177924736262),
        ("matched_tiny_target_count", 39),
        ("tiny_target_count", 39),
        ("unmatched_predicted_object_count", 5),
        ("threshold", DEFAULT_THRESHOLD),
    ):
        _require_equal(
            f"default profile metric {field}",
            metrics.get(field),
            expected,
        )

    split = payloads["d_run_split"]
    for field, expected in (
        ("dataset", DATASET),
        ("split_seed", SPLIT_SEED),
        ("used_train_count", TRAIN_COUNT),
        ("used_val_count", VALIDATION_COUNT),
        ("full_internal_train_count", TRAIN_COUNT),
        ("full_internal_val_count", VALIDATION_COUNT),
    ):
        _require_equal(f"split {field}", split.get(field), expected)
    split_hashes = split.get("hashes", {})
    for field, expected in (
        ("used_train_sha256", TRAIN_IDS_SHA256),
        ("full_internal_train_sha256", TRAIN_IDS_SHA256),
        ("used_val_sha256", VALIDATION_IDS_SHA256),
        ("full_internal_val_sha256", VALIDATION_IDS_SHA256),
    ):
        _require_equal(f"split hash {field}", split_hashes.get(field), expected)

    protocol = payloads["d_run_protocol"]
    _require_equal(
        "D protocol candidate variant",
        protocol.get("candidate_variant"),
        "tss_qfg",
    )
    _require_equal(
        "D protocol official-test boundary",
        protocol.get("official_test_accessed"),
        False,
    )
    normalization = protocol.get("normalization", {})
    _require_equal(
        "D normalization mean",
        normalization.get("mean"),
        NORMALIZATION_MEAN,
    )
    _require_equal(
        "D normalization standard deviation",
        normalization.get("std"),
        NORMALIZATION_STD,
    )
    formal = protocol.get("formal_contract", {})
    for field, expected in (
        ("dataset", DATASET),
        ("seed", TRAINING_SEED),
        ("split_seed", SPLIT_SEED),
        ("epochs", 800),
        ("threshold", DEFAULT_THRESHOLD),
        ("match_radius", MATCH_RADIUS),
        ("tiny_area", TINY_AREA),
        ("parent_checkpoint_sha256", PARENT_CHECKPOINT_SHA256),
        ("parent_checkpoint_epoch", 489),
        ("official_test_accessed", False),
    ):
        _require_equal(f"D formal contract {field}", formal.get(field), expected)

    sweep = payloads["d_selected_checkpoint_sweep"]
    for field, expected in (
        ("dataset", DATASET),
        ("variant", "tss_qfg"),
        ("seed", TRAINING_SEED),
        ("split_seed", SPLIT_SEED),
        ("validation_count", VALIDATION_COUNT),
        ("validation_split_sha256", VALIDATION_IDS_SHA256),
        ("checkpoint_role", "best_validation_miou_secondary"),
        ("checkpoint_epoch", 3),
        ("checkpoint_sha256", D_CHECKPOINT_SHA256),
        ("official_test_accessed", False),
    ):
        _require_equal(f"D sweep {field}", sweep.get(field), expected)
    evaluator = sweep.get("evaluator_contract", {})
    for field, expected in (
        ("fixed_threshold", DEFAULT_THRESHOLD),
        ("fa_budgets", list(FA_BUDGETS)),
        ("match_radius", MATCH_RADIUS),
        ("tiny_area", TINY_AREA),
        ("prediction_comparison", "prediction > threshold"),
        (
            "metric_core_sha256",
            FROZEN_SOURCE_SHA256["experiments/evaluate_pd_fa_sweep.py"],
        ),
        (
            "closed_interval_core_sha256",
            FROZEN_SOURCE_SHA256[
                "experiments/evaluate_tpd_clean_v6_pd_fa.py"
            ],
        ),
        ("official_test_accessed", False),
    ):
        _require_equal(f"evaluator contract {field}", evaluator.get(field), expected)
    source_binding = sweep.get("evaluation_source_binding", {})
    source_expectations = {
        "trainer": (
            "experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py"
        ),
        "shared_metric_core": "experiments/evaluate_pd_fa_sweep.py",
        "closed_interval_core": "experiments/evaluate_tpd_clean_v6_pd_fa.py",
        "evaluator": (
            "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py"
        ),
    }
    for role, relative in source_expectations.items():
        binding = source_binding.get(role, {})
        _require_equal(
            f"evaluation source {role} SHA",
            binding.get("sha256"),
            FROZEN_SOURCE_SHA256[relative],
        )

    training_lock = payloads["training_source_lock"]
    _require_equal(
        "training lock training-data SHA",
        training_lock.get("training_data_sha256"),
        TRAINING_DATA_SHA256,
    )
    parent = training_lock.get("parent_checkpoint", {})
    for field, expected in (
        ("sha256", PARENT_CHECKPOINT_SHA256),
        ("state_dict_sha256", PARENT_STATE_DICT_SHA256),
        ("epoch", 489),
    ):
        _require_equal(
            f"training lock parent {field}",
            parent.get(field),
            expected,
        )
    posttraining_lock = payloads["posttraining_source_lock"]
    _require_equal(
        "post-training/training source-lock binding",
        posttraining_lock.get("training_source_lock", {}).get("sha256"),
        records["training_source_lock"]["sha256"],
    )
    operational_lock = payloads["operational_source_lock_v2"]
    _require_equal(
        "operational/post-training source-lock binding",
        operational_lock.get(
            "upstream_posttraining_closure_source_lock", {}
        ).get("sha256"),
        records["posttraining_source_lock"]["sha256"],
    )

    reproducibility = payloads["reproducibility_manifest_v2"]
    _require_equal(
        "reproducibility status",
        reproducibility.get("status"),
        "complete",
    )
    project = reproducibility.get("project", {})
    _require_equal(
        "reproducibility scope",
        project.get("scope"),
        "single_seed_internal_validation",
    )
    _require_equal(
        "reproducibility official-test boundary",
        project.get("official_test_accessed"),
        False,
    )
    training = reproducibility.get("training_contract", {})
    _require_equal(
        "reproducibility independent child runs",
        training.get("runs_are_independent_after_common_initialization"),
        True,
    )
    _require_equal(
        "reproducibility parent initialization-only",
        training.get("parent_checkpoint", {}).get(
            "used_only_as_independent_run_initialization"
        ),
        True,
    )
    _require_equal(
        "reproducibility parent optimizer inheritance",
        training.get("parent_checkpoint", {}).get(
            "parent_optimizer_inherited"
        ),
        False,
    )
    operational = reproducibility.get("operational_default_v2", {})
    _require_equal(
        "reproducibility authoritative threshold",
        operational.get("selected_operating_point", {}).get("threshold"),
        DEFAULT_THRESHOLD,
    )
    _require_equal(
        "reproducibility deployment-v2 SHA",
        operational.get("deployment_manifest", {}).get("sha256"),
        records["deployment_manifest_v2"]["sha256"],
    )

    for role, payload in payloads.items():
        _assert_official_test_false(payload, role)


def build_parent_lock_payload(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Rebuild and validate the deterministic parent-lock payload."""

    root = _resolved_repo(repo_root)
    commit = _commit_binding(root)
    frozen_sources = _frozen_source_bindings(root)
    records, payloads = _validate_artifacts(root)
    _validate_source_locks(root, records, payloads)
    _validate_semantic_contracts(records, payloads)

    upstream_roles = (
        "deployment_manifest_v2",
        "default_operating_point_v2",
        "final_selection_v2",
        "d_run_split",
        "d_run_protocol",
        "d_selected_checkpoint_sweep",
        "seed42_factorial",
        "reproducibility_manifest_v2",
        "training_source_lock",
        "posttraining_source_lock",
        "operational_source_lock_v2",
    )
    payload = {
        "schema": LOCK_SCHEMA,
        "status": "complete",
        "lock_kind": LOCK_KIND,
        "candidate_family": "tpd_ner_v4_qfg_v2_croa",
        "frozen_model_source_commit": FROZEN_MODEL_SOURCE_COMMIT,
        "frozen_model_source": commit,
        "frozen_model_source_paths": frozen_sources,
        "upstream_authorities": {
            role: dict(records[role])
            for role in upstream_roles
        },
        "selected_model": {
            "method_id": "d_tss_qfg",
            "variant": "tss_qfg",
            "training_uses_tss": True,
            "inference_uses_tss": False,
            "d_training_checkpoint": {
                **records["d_training_checkpoint"],
                "checkpoint": "best_miou.pth.tar",
                "epoch": 3,
                "role": "best_validation_miou_secondary",
            },
            "final_inference_artifact": {
                **records["final_inference_artifact"],
                "strict_load": True,
                "survival_state_absent": True,
                "qfg_state_preserved": True,
                "parameter_count": 10870130,
                "state_key_count": 564,
            },
            "initialization_parent": {
                **records["initialization_parent_checkpoint"],
                "epoch": 489,
                "role": "best_validation_miou_secondary",
                "state_dict_sha256": PARENT_STATE_DICT_SHA256,
                "usage": "initialization_only",
                "copied_shared_state_key_count": 544,
                "parent_optimizer_inherited": False,
                "parent_model_trained_with_child": False,
                "child_trainable_scope": "all_model_parameters",
                "children_independent_after_common_initialization": True,
            },
            "operating_point": {
                "authority": "deployment_v2_authoritative_default",
                "source": "fixed_threshold_0_5",
                "threshold": DEFAULT_THRESHOLD,
                "metrics": {
                    "matched_target_count": 188,
                    "target_count": 189,
                    "pd": 188 / 189,
                    "fa": 4.1301985432330825e-6,
                    "miou": 0.9370177924736262,
                    "matched_tiny_target_count": 39,
                    "tiny_target_count": 39,
                    "tiny_pd": 1.0,
                    "false_objects": 5,
                },
            },
        },
        "arm_training_semantics": {
            "original": {
                "variant": "original",
                "common_v4_parent_used": False,
                "independent_complete_model_training": True,
            },
            "a_v4_stack_control": {
                "variant": "tss_control",
                "common_v4_parent_used_for_initialization_only": True,
                "tss_weight": 0.0,
                "qfg_enabled": False,
            },
            "b_v4_stack_tss": {
                "variant": "tss_on",
                "common_v4_parent_used_for_initialization_only": True,
                "tss_weight": 0.005,
                "qfg_enabled": False,
            },
            "c_v4_stack_qfg": {
                "variant": "qfg_only",
                "common_v4_parent_used_for_initialization_only": True,
                "tss_weight": 0.0,
                "qfg_enabled": True,
            },
            "d_full_stack": {
                "variant": "tss_qfg",
                "common_v4_parent_used_for_initialization_only": True,
                "tss_weight": 0.005,
                "qfg_enabled": True,
            },
            "shared_rule": {
                "one_independent_child_instance_per_arm": True,
                "one_new_optimizer_per_child": True,
                "all_child_parameters_trainable": True,
                "parent_optimizer_scheduler_and_epoch_inherited": False,
                "cross_arm_resume_forbidden": True,
            },
        },
        "data_contract": {
            "dataset": DATASET,
            "scope": "official_training_set_internal_split",
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "train_count": TRAIN_COUNT,
            "validation_count": VALIDATION_COUNT,
            "train_ids_sha256": TRAIN_IDS_SHA256,
            "validation_ids_sha256": VALIDATION_IDS_SHA256,
            "training_data_sha256": TRAINING_DATA_SHA256,
            "normalization": {
                "mean": NORMALIZATION_MEAN,
                "std": NORMALIZATION_STD,
            },
            "official_test_accessed": False,
        },
        "evaluator_contract": {
            "prediction_comparison": "prediction > threshold",
            "fixed_threshold": DEFAULT_THRESHOLD,
            "fa_budgets": list(FA_BUDGETS),
            "match_radius": MATCH_RADIUS,
            "tiny_area": TINY_AREA,
            "primary_metrics": [
                "pd",
                "fa",
                "miou",
                "tiny_pd",
                "false_objects",
            ],
            "source_sha256": {
                relative: digest
                for relative, digest in sorted(
                    FROZEN_SOURCE_SHA256.items()
                )
                if "evaluate_" in relative
            },
            "official_test_accessed": False,
        },
        "claim_boundary": {
            "scope": "single_seed_internal_validation",
            "final_model_engineering_selected": True,
            "final_model_established": True,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "cross_seed_stability_claim": False,
            "official_test_claim": False,
            "official_test_accessed": False,
            "internal_validation_only": True,
        },
        "certification_state": {
            "certification_design_reviewed": True,
            "certification_design_complete": False,
            "certification_implementation_complete": False,
            "certification_execution_authorized": False,
        },
        "policy": {
            "canonical_pretty_json": True,
            "one_trailing_newline": True,
            "write_once": True,
            "overwrite_forbidden": True,
            "repo_relative_paths_only": True,
            "regular_non_symlink_inputs": True,
            "live_source_maps_verified": True,
            "frozen_commit_blobs_match_live_sources": True,
            "parent_lock_self_excluded": True,
            "freezer_self_excluded": True,
            "certification_protocol_self_excluded": True,
            "certification_commit_bound": False,
            "certification_commit_deferred_to_release_attestation": True,
            "official_test_accessed": False,
        },
    }
    return json.loads(canonical_json_bytes(payload))


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _output_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return Path(os.path.abspath(raw))


def _require_new_output(path: Path) -> Path:
    output = _output_path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing certification parent lock: {output}"
        )
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise NotADirectoryError(
            f"parent-lock directory must already exist: {parent}"
        )
    return output


def _write_new_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    output = _require_new_output(path)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.write-once.tmp"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary parent-lock path exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(output.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return output


def verify_parent_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    output = _output_path(path)
    if output.is_symlink() or not output.is_file():
        raise ValueError(
            f"certification parent lock must be a regular file: {output}"
        )
    raw, observed = _load_json_file(output, "certification parent lock")
    _require_equal(
        "certification parent-lock canonical bytes",
        raw,
        canonical_json_bytes(observed),
    )
    expected = build_parent_lock_payload(repo_root)
    _require_equal(
        "certification parent-lock live payload",
        observed,
        expected,
    )
    _require_equal(
        "certification parent-lock live bytes",
        raw,
        canonical_json_bytes(expected),
    )
    return observed


def plan_parent_lock(
    *,
    repo_root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _require_new_output(output)
    payload = build_parent_lock_payload(repo_root)
    return {
        "schema": PLAN_SCHEMA,
        "status": "ready",
        "action": "plan",
        "output": str(destination),
        "output_exists": False,
        "would_write": False,
        "overwrite_forbidden": True,
        "payload_sha256": payload_sha256(payload),
        "upstream_authority_count": len(payload["upstream_authorities"]),
        "frozen_source_count": len(payload["frozen_model_source_paths"]),
    }


def write_parent_lock_once(
    *,
    repo_root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _require_new_output(output)
    payload = build_parent_lock_payload(repo_root)
    _write_new_atomic(destination, payload)
    verified = verify_parent_lock(destination, repo_root=repo_root)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "write-once",
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "payload_sha256": payload_sha256(verified),
        "post_write_verified": True,
        "overwrite_forbidden": True,
        "upstream_authority_count": len(verified["upstream_authorities"]),
        "frozen_source_count": len(verified["frozen_model_source_paths"]),
    }


def verify_parent_lock_action(
    *,
    repo_root: Path = REPO_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _output_path(output)
    payload = verify_parent_lock(destination, repo_root=repo_root)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "payload_sha256": payload_sha256(payload),
        "verified": True,
        "upstream_authority_count": len(payload["upstream_authorities"]),
        "frozen_source_count": len(payload["frozen_model_source_paths"]),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plan:
        result = plan_parent_lock(
            repo_root=args.repo_root,
            output=args.output,
        )
    elif args.write_once:
        result = write_parent_lock_once(
            repo_root=args.repo_root,
            output=args.output,
        )
    else:
        result = verify_parent_lock_action(
            repo_root=args.repo_root,
            output=args.output,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "ARTIFACT_CONTRACTS",
    "DEFAULT_OUTPUT",
    "DEFAULT_OUTPUT_RELATIVE_PATH",
    "FROZEN_MODEL_SOURCE_COMMIT",
    "FROZEN_SOURCE_SHA256",
    "LOCK_KIND",
    "LOCK_SCHEMA",
    "PLAN_SCHEMA",
    "build_parent_lock_payload",
    "canonical_json_bytes",
    "main",
    "parse_args",
    "payload_sha256",
    "plan_parent_lock",
    "sha256_file",
    "verify_parent_lock",
    "verify_parent_lock_action",
    "write_parent_lock_once",
]


if __name__ == "__main__":
    main()
