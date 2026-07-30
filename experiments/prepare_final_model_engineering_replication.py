#!/usr/bin/env python3
"""Create/verify the seed contract and four engineering B/D child manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import final_model_child_initialization_manifest as child  # noqa: E402
from experiments import final_model_replication_seed_contract as seeds  # noqa: E402
from experiments import freeze_final_model_certification_source_lock as source_lock  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as b_trainer,
)


DEFAULT_SEED_CONTRACT = (
    REPO_ROOT / "experiments/final_model_replication_seed_contract.json"
)
DEFAULT_MANIFEST_DIRECTORY = (
    REPO_ROOT / "experiments/final_model_replication_manifests_v1"
)


def manifest_path(
    directory: Path,
    *,
    seed: int,
    arm: str,
) -> Path:
    return Path(directory) / f"seed_{seed}_{arm}_child_init.json"


def prepare(
    *,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    seed_contract_path: Path = DEFAULT_SEED_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
) -> dict[str, Any]:
    source_lock.verify_source_lock(source_lock_path, repo_root=REPO_ROOT)
    source_sha256 = source_lock.sha256_file(source_lock_path)
    schedule = seeds.ReplicationSeedScheduleContract.derive(source_sha256)
    seeds.write_contract_once(seed_contract_path, schedule)
    seed_contract_sha256 = seeds.file_sha256(seed_contract_path)
    manifests: list[dict[str, Any]] = []
    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        for arm in child.SUPPORTED_ARMS:
            manifest = child.ChildInitializationManifest(
                arm=arm,
                trajectory_seed=trajectory_seed,
                seed_contract_sha256=seed_contract_sha256,
                certification_source_lock_sha256=source_sha256,
                parent_seed=seeds.BUILDER_COMPATIBILITY_SEED,
                parent_checkpoint_path=str(
                    b_trainer.PARENT_CHECKPOINT_PATH.resolve()
                ),
                parent_checkpoint_sha256=(
                    b_trainer.PARENT_CHECKPOINT_SHA256
                ),
                parent_state_dict_sha256=(
                    b_trainer.PARENT_STATE_DICT_SHA256
                ),
                parent_checkpoint_epoch=b_trainer.PARENT_CHECKPOINT_EPOCH,
            )
            destination = manifest_path(
                manifest_directory,
                seed=trajectory_seed,
                arm=arm,
            )
            child.write_manifest_once(destination, manifest)
            manifests.append(
                {
                    "arm": arm,
                    "trajectory_seed": trajectory_seed,
                    "path": str(destination.resolve()),
                    "sha256": child.file_sha256(destination),
                }
            )
    return {
        "schema": "sctransnet_final_model_engineering_replication_prepare_v1",
        "status": "complete",
        "source_lock_path": str(source_lock_path.resolve()),
        "source_lock_sha256": source_sha256,
        "seed_contract_path": str(seed_contract_path.resolve()),
        "seed_contract_sha256": seed_contract_sha256,
        "engineering_trajectory_seeds": list(
            seeds.ENGINEERING_TRAJECTORY_SEEDS
        ),
        "manifests": manifests,
        "run_count": len(manifests),
        "confirmatory_training_authorized": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=DEFAULT_SEED_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=DEFAULT_MANIFEST_DIRECTORY,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = prepare(
        source_lock_path=args.source_lock,
        seed_contract_path=args.seed_contract,
        manifest_directory=args.manifest_directory,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
