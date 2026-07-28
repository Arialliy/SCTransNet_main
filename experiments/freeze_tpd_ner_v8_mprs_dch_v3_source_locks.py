#!/usr/bin/env python3
"""Create or verify the independent V3 NER source manifests.

Importing this module never publishes a lock.  ``freeze`` is an explicit,
no-overwrite operation that is valid only after the complete V3 training and
acceptance source closures have been finalized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v2_source_locks as v2_freeze,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as exact,
)
from experiments.freeze_tpd_clean_v8_mprs_dch_source_locks import (  # noqa: E402
    file_sha256,
    hash_sources,
    load_json_object,
    publish_new_lock,
    training_data_contract,
)


TRAINING_SCHEMA = exact.EXACT_SOURCE_LOCK_SCHEMA
ACCEPTANCE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_acceptance_source_lock_v1"
)
CANDIDATE_FAMILY = exact.CANDIDATE_FAMILY
DATASET = "NUDT-SIRST"
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
VARIANTS = exact.supported_candidate_variants()
V1_CONTROL = exact.V8_PARENT_RELAY_OFF_REFERENCE
V2_PREDECESSOR = exact.V2_RELAY_ON_VARIANT
DEFAULT_TRAINING_LOCK = exact.DEFAULT_EXACT_SOURCE_LOCK_PATH
DEFAULT_ACCEPTANCE_LOCK = (
    REPO_ROOT
    / "experiments/tpd_ner_v8_mprs_dch_v3_acceptance_source_lock.json"
)
UPSTREAM_V2_TRAINING_LOCK = v2_freeze.DEFAULT_TRAINING_LOCK
UPSTREAM_V2_ACCEPTANCE_LOCK = v2_freeze.DEFAULT_ACCEPTANCE_LOCK

# This list intentionally names the complete future production closure.  A
# missing file keeps ``freeze`` from succeeding; tests may supply a smaller
# explicit source set when exercising the pure lock-building functions.
ACCEPTANCE_SOURCE_RELATIVES = (
    "experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_v3_formal800.py",
    "experiments/smoke_tpd_ner_v8_mprs_dch_v3.py",
    "experiments/handoff_tpd_ner_v8_v2_to_v3.py",
    "experiments/launch_tpd_ner_v8_mprs_dch_v3_formal800_1x5090.sh",
    "experiments/run_tpd_ner_v8_mprs_dch_v3_formal800_1x5090_lane.sh",
    "experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_v2_formal800.py",
    "experiments/handoff_tpd_ner_v8_v1_to_v2.py",
    "experiments/freeze_tpd_ner_v8_mprs_dch_v2_source_locks.py",
    "experiments/postprocess_tpd_ner_v8_mprs_dch_formal800.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_sctransnet_baseline_reference_closed_interval.py",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
)


def training_source_relatives() -> tuple[str, ...]:
    root = exact.REPO_ROOT.resolve()
    relatives = tuple(
        str(path.resolve().relative_to(root))
        for path in exact.RUNTIME_SOURCE_PATHS
    )
    if len(relatives) != len(set(relatives)):
        raise RuntimeError("V3 exact runtime source list contains duplicates")
    required = {
        "experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py",
        "experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md",
        "model/tpd_ner_v8_mprs_dch_v3.py",
        "model/tpd_ner_v8_mprs_dch_v2.py",
        "experiments/train_tpd_ner_v8_mprs_dch_exact.py",
        "model/tpd_ner_v8_mprs_dch.py",
    }
    missing = sorted(required - set(relatives))
    if missing:
        raise RuntimeError(f"V3 runtime sources omit dependencies: {missing}")
    return relatives


def formal_contract() -> dict[str, Any]:
    contract = dict(exact.formal_contract())
    if tuple(contract.get("candidate_variants", ())) != VARIANTS:
        raise ValueError("V3 exact formal variant matrix differs")
    if VARIANTS != (
        exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
    ):
        raise ValueError("V3 training manifest must contain only relay-on")
    if (
        contract.get("required_control") != V1_CONTROL
        or contract.get("structural_predecessor") != V2_PREDECESSOR
        or contract.get("relay_off_retrained") is not False
    ):
        raise ValueError("V3 exact comparison contract differs")
    return contract


def performance_gate_contract() -> dict[str, Any]:
    paired_common = {
        "minimum_non_inferior_budget_count": 4,
        "minimum_strictly_better_budget_count": 1,
        "budget_count": 5,
    }
    return {
        "anchor_target_count": 189,
        "pd_primary_fixed_threshold_0_5": {
            "minimum_matched_targets": 188,
            "minimum_pd": 188 / 189,
            "maximum_fa": 1e-6,
            "minimum_miou": 0.933647,
        },
        "miou_secondary_fixed_threshold_0_5": {
            "minimum_matched_targets": 187,
            "minimum_pd": 187 / 189,
            "maximum_fa": 1e-6,
            "minimum_miou": 0.946542,
        },
        "pd_at_fa_budget": {
            "1e-06": {
                "minimum_matched_targets": 187,
                "minimum_pd": 187 / 189,
            },
            **{
                f"{budget:.10g}": {
                    "minimum_matched_targets": 188,
                    "minimum_pd": 188 / 189,
                }
                for budget in FA_BUDGETS[1:]
            },
        },
        "paired_v3_on_vs_v1_off_each_checkpoint_role": {
            "reference": V1_CONTROL,
            "candidate": (
                exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
            ),
            **paired_common,
        },
        "paired_v3_on_vs_v2_on_each_checkpoint_role": {
            "reference": V2_PREDECESSOR,
            "candidate": (
                exact.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON
            ),
            **paired_common,
        },
        "all_required_components": [
            "pd_primary_absolute",
            "miou_secondary_absolute",
            "pd_primary_v3_vs_v1",
            "miou_secondary_v3_vs_v1",
            "pd_primary_v3_vs_v2",
            "miou_secondary_v3_vs_v2",
        ],
        "tiny_pd_reported_not_independent_gate": True,
        "v1_off_absolute_gate_required": False,
        "v2_predecessor_absolute_gate_required": False,
        "baseline_affects_decision": False,
    }


def training_policy_contract() -> dict[str, Any]:
    return {
        "official_test_accessed": False,
        "physical_gpu_choices": [2, 3],
        "simultaneous_v3_training_tasks": 1,
        "gpu0_gpu1_used": False,
        "training_seed": 42,
        "split_seed": 20260722,
        "multi_seed_scheduled": False,
        "required_control": V1_CONTROL,
        "structural_predecessor": V2_PREDECESSOR,
        "relay_off_retrained": False,
        "fresh_or_exact_resume_only": True,
        "cross_version_exact_resume_supported": False,
        "existing_manifest_overwrite_forbidden": True,
        "source_symlinks_forbidden": True,
    }


def acceptance_policy_contract() -> dict[str, Any]:
    return {
        "official_test_accessed": False,
        "training_seed": 42,
        "split_seed": 20260722,
        "multi_seed_scheduled": False,
        "new_sweeps": [
            (
                "tpd_ner_v8_mprs_dch_v3_full_relay_on/"
                "best.pth.tar"
            ),
            (
                "tpd_ner_v8_mprs_dch_v3_full_relay_on/"
                "best_miou.pth.tar"
            ),
        ],
        "v1_off_sweeps_read_only": True,
        "v2_predecessor_sweeps_read_only": True,
        "baseline_sweeps_read_only": True,
        "all_gate_components_required": True,
        "existing_manifest_overwrite_forbidden": True,
        "source_symlinks_forbidden": True,
    }


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
        else Path(dataset_dir)
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
        "policy": training_policy_contract(),
    }


def _verified_v2_upstream(
    *,
    repo_root: Path,
    dataset_dir: Path | None,
    training_lock: Path,
    acceptance_lock: Path,
) -> dict[str, str]:
    v2_training = v2_freeze.verify_training_lock(
        training_lock,
        repo_root=repo_root,
        dataset_dir=(
            repo_root.resolve() / "datasets"
            if dataset_dir is None
            else dataset_dir
        ),
    )
    v2_freeze.verify_acceptance_lock(
        acceptance_lock,
        training_lock,
        repo_root=repo_root,
    )
    return {
        "upstream_v2_training_source_lock_sha256": file_sha256(
            training_lock
        ),
        "upstream_v2_acceptance_source_lock_sha256": file_sha256(
            acceptance_lock
        ),
        "upstream_v2_training_data_sha256": str(
            v2_training["training_data_sha256"]
        ),
    }


def build_acceptance_lock(
    training_lock_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path | None = None,
    source_relatives: Sequence[str] = ACCEPTANCE_SOURCE_RELATIVES,
    expected_training_source_relatives: Sequence[str] | None = None,
    expected_training_contract: Mapping[str, Any] | None = None,
    upstream_v2_training_lock: Path = UPSTREAM_V2_TRAINING_LOCK,
    upstream_v2_acceptance_lock: Path = UPSTREAM_V2_ACCEPTANCE_LOCK,
) -> dict[str, Any]:
    training = verify_training_lock(
        training_lock_path,
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        expected_source_relatives=expected_training_source_relatives,
        expected_contract=expected_training_contract,
    )
    upstream = _verified_v2_upstream(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        training_lock=upstream_v2_training_lock,
        acceptance_lock=upstream_v2_acceptance_lock,
    )
    if (
        upstream["upstream_v2_training_data_sha256"]
        != training.get("training_data_sha256")
    ):
        raise ValueError("V3/V2 training data binding differs")
    sources = hash_sources(repo_root, source_relatives)
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "lock_kind": "acceptance",
        "candidate_family": CANDIDATE_FAMILY,
        "dataset": DATASET,
        "variants": list(VARIANTS),
        "required_control": V1_CONTROL,
        "structural_predecessor": V2_PREDECESSOR,
        "relay_off_retrained": False,
        "training_source_lock_sha256": file_sha256(training_lock_path),
        "training_data_sha256": training.get("training_data_sha256"),
        **upstream,
        "performance_gate_contract": performance_gate_contract(),
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": acceptance_policy_contract(),
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
        raise ValueError("V3 training source-manifest identity differs")
    sources = (
        training_source_relatives()
        if expected_source_relatives is None
        else tuple(expected_source_relatives)
    )
    if set(payload.get("source_sha256", ())) != set(sources):
        raise ValueError("V3 training runtime source set differs")
    contract = (
        formal_contract()
        if expected_contract is None
        else dict(expected_contract)
    )
    if payload.get("formal_contract") != contract:
        raise ValueError("V3 training formal contract differs")
    live_data = training_data_contract(
        repo_root.resolve() / "datasets"
        if dataset_dir is None
        else dataset_dir,
        DATASET,
    )
    for field, expected in live_data.items():
        if payload.get(field) != expected:
            raise ValueError(f"V3 training data contract differs: {field}")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("V3 training policy is missing")
    expected_policy = training_policy_contract()
    if set(policy) != set(expected_policy):
        raise ValueError("V3 training policy field set differs")
    for name, expected in expected_policy.items():
        if policy.get(name) != expected:
            raise ValueError(f"V3 training policy differs: {name}")
    _verify_source_mapping(payload, repo_root=repo_root)
    return payload


def verify_acceptance_lock(
    path: Path,
    training_lock_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    dataset_dir: Path | None = None,
    expected_source_relatives: Sequence[str] = ACCEPTANCE_SOURCE_RELATIVES,
    upstream_v2_training_lock: Path = UPSTREAM_V2_TRAINING_LOCK,
    upstream_v2_acceptance_lock: Path = UPSTREAM_V2_ACCEPTANCE_LOCK,
) -> dict[str, Any]:
    payload = load_json_object(path)
    if (
        payload.get("schema") != ACCEPTANCE_SCHEMA
        or payload.get("lock_kind") != "acceptance"
        or payload.get("candidate_family") != CANDIDATE_FAMILY
        or payload.get("dataset") != DATASET
        or tuple(payload.get("variants", ())) != VARIANTS
        or payload.get("required_control") != V1_CONTROL
        or payload.get("structural_predecessor") != V2_PREDECESSOR
        or payload.get("relay_off_retrained") is not False
    ):
        raise ValueError("V3 acceptance source-manifest identity differs")
    training = verify_training_lock(
        training_lock_path,
        repo_root=repo_root,
        dataset_dir=dataset_dir,
    )
    if (
        payload.get("training_source_lock_sha256")
        != file_sha256(training_lock_path)
        or payload.get("training_data_sha256")
        != training.get("training_data_sha256")
    ):
        raise ValueError("V3 acceptance training binding differs")
    upstream = _verified_v2_upstream(
        repo_root=repo_root,
        dataset_dir=dataset_dir,
        training_lock=upstream_v2_training_lock,
        acceptance_lock=upstream_v2_acceptance_lock,
    )
    if (
        upstream["upstream_v2_training_data_sha256"]
        != training.get("training_data_sha256")
    ):
        raise ValueError("V3/V2 training data binding differs")
    for name, expected in upstream.items():
        if payload.get(name) != expected:
            raise ValueError(f"V3 acceptance upstream binding differs: {name}")
    if payload.get("performance_gate_contract") != performance_gate_contract():
        raise ValueError("V3 acceptance performance gates differ")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("V3 acceptance policy is missing")
    expected_policy = acceptance_policy_contract()
    if set(policy) != set(expected_policy):
        raise ValueError("V3 acceptance policy field set differs")
    for name, expected in expected_policy.items():
        if policy.get(name) != expected:
            raise ValueError(f"V3 acceptance policy differs: {name}")
    if set(payload.get("source_sha256", ())) != set(
        expected_source_relatives
    ):
        raise ValueError("V3 acceptance source set differs")
    _verify_source_mapping(payload, repo_root=repo_root)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify V3 five-node NER source manifests"
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
    parser.add_argument(
        "--upstream-v2-training-lock",
        type=Path,
        default=UPSTREAM_V2_TRAINING_LOCK,
    )
    parser.add_argument(
        "--upstream-v2-acceptance-lock",
        type=Path,
        default=UPSTREAM_V2_ACCEPTANCE_LOCK,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    common = {
        "repo_root": REPO_ROOT,
        "dataset_dir": args.dataset_dir,
    }
    if args.mode == "freeze":
        if args.kind in ("training", "all"):
            publish_new_lock(
                args.training_lock,
                build_training_lock(dataset_dir=args.dataset_dir),
            )
        if args.kind in ("acceptance", "all"):
            publish_new_lock(
                args.acceptance_lock,
                build_acceptance_lock(
                    args.training_lock,
                    dataset_dir=args.dataset_dir,
                    upstream_v2_training_lock=(
                        args.upstream_v2_training_lock
                    ),
                    upstream_v2_acceptance_lock=(
                        args.upstream_v2_acceptance_lock
                    ),
                ),
            )
    else:
        if args.kind in ("training", "all"):
            verify_training_lock(args.training_lock, **common)
        if args.kind in ("acceptance", "all"):
            verify_acceptance_lock(
                args.acceptance_lock,
                args.training_lock,
                **common,
                upstream_v2_training_lock=(
                    args.upstream_v2_training_lock
                ),
                upstream_v2_acceptance_lock=(
                    args.upstream_v2_acceptance_lock
                ),
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
