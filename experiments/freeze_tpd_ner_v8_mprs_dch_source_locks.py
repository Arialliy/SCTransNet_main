#!/usr/bin/env python3
"""Create or verify the V8-MPRS-DCH + five-node NER source manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_ner_v8_mprs_dch_exact as exact  # noqa: E402
from experiments.freeze_tpd_clean_v8_mprs_dch_source_locks import (  # noqa: E402
    file_sha256,
    hash_sources,
    load_json_object,
    publish_new_lock,
    training_data_contract,
)


TRAINING_SCHEMA = exact.EXACT_SOURCE_LOCK_SCHEMA
ACCEPTANCE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_acceptance_source_lock_v1"
)
CANDIDATE_FAMILY = "tpd_ner_v8_mprs_dch"
DATASET = "NUDT-SIRST"
VARIANTS = exact.supported_candidate_variants()
DEFAULT_TRAINING_LOCK = exact.DEFAULT_EXACT_SOURCE_LOCK_PATH
DEFAULT_ACCEPTANCE_LOCK = (
    REPO_ROOT / "experiments/tpd_ner_v8_mprs_dch_acceptance_source_lock.json"
)
ACCEPTANCE_SOURCE_RELATIVES = (
    "experiments/TPD_NER_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/smoke_tpd_ner_v8_mprs_dch.py",
    "experiments/launch_tpd_ner_v8_mprs_dch_formal800_2x5090.sh",
    "experiments/run_tpd_ner_v8_mprs_dch_formal800_2x5090_lane.sh",
    "experiments/freeze_tpd_ner_v8_mprs_dch_source_locks.py",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
)


def training_source_relatives() -> tuple[str, ...]:
    root = exact.REPO_ROOT.resolve()
    relatives = tuple(
        str(path.resolve().relative_to(root))
        for path in exact.RUNTIME_SOURCE_PATHS
    )
    if len(relatives) != len(set(relatives)):
        raise RuntimeError("NER exact runtime source list contains duplicates")
    return relatives


def formal_contract() -> dict[str, Any]:
    return dict(exact.formal_contract())


def build_training_lock(
    *,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path | None = None,
    source_relatives: Sequence[str] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_dir = (
        repo_root.resolve() / "datasets"
        if dataset_dir is None
        else dataset_dir
    )
    relatives = (
        training_source_relatives()
        if source_relatives is None
        else tuple(source_relatives)
    )
    frozen_contract = (
        formal_contract() if contract is None else dict(contract)
    )
    data = training_data_contract(dataset_dir, DATASET)
    sources = hash_sources(repo_root, relatives)
    return {
        "schema": TRAINING_SCHEMA,
        "lock_kind": "training",
        "candidate_family": CANDIDATE_FAMILY,
        **data,
        "variants": list(VARIANTS),
        "formal_contract": frozen_contract,
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": {
            "official_test_accessed": False,
            "physical_gpus": [2, 3],
            "gpu0_gpu1_used": False,
            "training_seed": 42,
            "split_seed": 20260722,
            "multi_seed_scheduled": False,
            "fresh_or_exact_resume_only": True,
            "cross_variant_exact_resume_forbidden": True,
            "existing_manifest_overwrite_forbidden": True,
            "source_symlinks_forbidden": True,
        },
    }


def build_acceptance_lock(
    training_lock_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    source_relatives: Sequence[str] = ACCEPTANCE_SOURCE_RELATIVES,
) -> dict[str, Any]:
    training = load_json_object(training_lock_path)
    if training.get("schema") != TRAINING_SCHEMA:
        raise ValueError("training source-manifest schema mismatch")
    sources = hash_sources(repo_root, source_relatives)
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "lock_kind": "acceptance",
        "candidate_family": CANDIDATE_FAMILY,
        "dataset": DATASET,
        "variants": list(VARIANTS),
        "training_source_lock_sha256": file_sha256(training_lock_path),
        "training_data_sha256": training.get("training_data_sha256"),
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": {
            "official_test_accessed": False,
            "training_seed": 42,
            "split_seed": 20260722,
            "multi_seed_scheduled": False,
            "performance_metrics": [
                "Pd",
                "Fa",
                "mIoU",
                "false_objects_per_image",
                "Pd_at_Fa_budget",
            ],
            "formal_checkpoints": [
                "best.pth.tar",
                "best_miou.pth.tar",
            ],
            "existing_manifest_overwrite_forbidden": True,
            "source_symlinks_forbidden": True,
        },
    }


def _verify_source_mapping(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source manifest has no source_sha256 mapping")
    if payload.get("source_count") != len(sources):
        raise ValueError("source_count differs from source_sha256 mapping")
    expected = hash_sources(repo_root, tuple(sources))
    if expected != sources:
        changed = sorted(
            relative
            for relative in set(expected) | set(sources)
            if expected.get(relative) != sources.get(relative)
        )
        raise ValueError(f"source digests differ: {changed}")


def verify_training_lock(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path | None = None,
    expected_source_relatives: Sequence[str] | None = None,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_json_object(path)
    if (
        payload.get("schema") != TRAINING_SCHEMA
        or payload.get("lock_kind") != "training"
        or payload.get("candidate_family") != CANDIDATE_FAMILY
        or tuple(payload.get("variants", ())) != VARIANTS
    ):
        raise ValueError("training source-manifest identity differs")
    sources = (
        training_source_relatives()
        if expected_source_relatives is None
        else tuple(expected_source_relatives)
    )
    if set(payload.get("source_sha256", ())) != set(sources):
        raise ValueError("training runtime source set differs")
    contract = (
        formal_contract()
        if expected_contract is None
        else dict(expected_contract)
    )
    if payload.get("formal_contract") != contract:
        raise ValueError("training formal contract differs")
    live_data = training_data_contract(
        repo_root.resolve() / "datasets"
        if dataset_dir is None
        else dataset_dir,
        DATASET,
    )
    for field, expected in live_data.items():
        if payload.get(field) != expected:
            raise ValueError(f"training data contract differs: {field}")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("training policy is missing")
    expected_policy = {
        "training_seed": 42,
        "split_seed": 20260722,
        "multi_seed_scheduled": False,
        "physical_gpus": [2, 3],
        "gpu0_gpu1_used": False,
    }
    for name, expected in expected_policy.items():
        if policy.get(name) != expected:
            raise ValueError(f"training policy differs: {name}")
    _verify_source_mapping(payload, repo_root=repo_root)
    return payload


def verify_acceptance_lock(
    path: Path,
    training_lock_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    expected_source_relatives: Sequence[str] = ACCEPTANCE_SOURCE_RELATIVES,
) -> dict[str, Any]:
    payload = load_json_object(path)
    if (
        payload.get("schema") != ACCEPTANCE_SCHEMA
        or payload.get("lock_kind") != "acceptance"
        or payload.get("candidate_family") != CANDIDATE_FAMILY
        or tuple(payload.get("variants", ())) != VARIANTS
    ):
        raise ValueError("acceptance source-manifest identity differs")
    training = load_json_object(training_lock_path)
    if training.get("schema") != TRAINING_SCHEMA:
        raise ValueError("bound training source-manifest schema differs")
    if (
        payload.get("training_source_lock_sha256")
        != file_sha256(training_lock_path)
        or payload.get("training_data_sha256")
        != training.get("training_data_sha256")
    ):
        raise ValueError("acceptance manifest training binding differs")
    if set(payload.get("source_sha256", ())) != set(
        expected_source_relatives
    ):
        raise ValueError("acceptance source set differs")
    _verify_source_mapping(payload, repo_root=repo_root)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify V8-MPRS-DCH five-node NER manifests"
    )
    parser.add_argument(
        "--mode",
        choices=("freeze", "verify"),
        required=True,
    )
    parser.add_argument(
        "--kind",
        choices=("training", "acceptance", "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
    )
    parser.add_argument(
        "--training-lock",
        type=Path,
        default=DEFAULT_TRAINING_LOCK,
    )
    parser.add_argument(
        "--acceptance-lock",
        type=Path,
        default=DEFAULT_ACCEPTANCE_LOCK,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "freeze":
        if args.kind in ("training", "all"):
            publish_new_lock(
                args.training_lock,
                build_training_lock(dataset_dir=args.dataset_dir),
            )
        if args.kind in ("acceptance", "all"):
            publish_new_lock(
                args.acceptance_lock,
                build_acceptance_lock(args.training_lock),
            )
    else:
        if args.kind in ("training", "all"):
            verify_training_lock(
                args.training_lock,
                dataset_dir=args.dataset_dir,
            )
        if args.kind in ("acceptance", "all"):
            verify_acceptance_lock(
                args.acceptance_lock,
                args.training_lock,
            )
    output: dict[str, Any] = {
        "status": "complete",
        "mode": args.mode,
        "kind": args.kind,
    }
    if args.training_lock.is_file():
        output["training_source_lock_sha256"] = file_sha256(
            args.training_lock
        )
    if args.acceptance_lock.is_file():
        output["acceptance_source_lock_sha256"] = file_sha256(
            args.acceptance_lock
        )
    print(json.dumps(output, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
