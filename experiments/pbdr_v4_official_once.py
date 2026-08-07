"""Exclusive-claim, one-loader, one-pass boundary for PBDR-V4 official data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from experiments.pbdr_v4_candidate_pool import validate_candidate_pool


OFFICIAL_CLAIM_SCHEMA = "sctransnet_pbdr_v4_official_claim/v1"
OFFICIAL_BUNDLE_SCHEMA = "sctransnet_pbdr_v4_official_bundle/v1"
OFFICIAL_FAILURE_SCHEMA = "sctransnet_pbdr_v4_official_consumed_failure/v1"
JOINT_CANDIDATE_POOL_SCHEMA = "sctransnet_pbdr_v4_joint_candidate_pool/v1"
JOINT_ROLE_ORDER = ("best_miou", "best_pd")


class PBDRV4OfficialOnceError(RuntimeError):
    """The official one-pass boundary is consumed or an artifact is invalid."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4OfficialOnceError(f"official artifact is not JSON-safe: {error}") from error


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exclusive_json(path: Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise PBDRV4OfficialOnceError(f"exclusive official artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise PBDRV4OfficialOnceError("official artifact parent is a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(dict(payload)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4OfficialOnceError(f"{label} is missing or unsafe")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4OfficialOnceError(f"cannot read {label}: {error}") from error
    if not isinstance(value, Mapping):
        raise PBDRV4OfficialOnceError(f"{label} must contain one object")
    return dict(value)


def _validate_bundle(
    bundle: Mapping[str, object],
    *,
    candidate_pool: Mapping[str, object],
) -> dict[str, object]:
    if bundle.get("schema") != OFFICIAL_BUNDLE_SCHEMA or bundle.get("status") != "committed":
        raise PBDRV4OfficialOnceError("official bundle identity/status differs")
    if bundle.get("candidate_pool_sha256") != candidate_pool.get("candidate_pool_sha256"):
        raise PBDRV4OfficialOnceError("official bundle candidate-pool binding differs")
    declared = bundle.get("bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256", None)
    if declared != _canonical_sha(unsigned):
        raise PBDRV4OfficialOnceError("official bundle canonical SHA differs")
    sample_count = bundle.get("sample_count")
    counts = bundle.get("forward_counts")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise PBDRV4OfficialOnceError("official bundle sample count differs")
    if not isinstance(counts, Mapping):
        raise PBDRV4OfficialOnceError("official bundle forward counts differ")
    families = candidate_pool.get("family_order")
    if not isinstance(families, list) or set(counts) != set(families) or any(
        counts[family] != sample_count for family in families
    ):
        raise PBDRV4OfficialOnceError("official candidate forward count differs")
    if bundle.get("loader_iteration_count") != 1:
        raise PBDRV4OfficialOnceError("official loader iteration count differs")
    return dict(bundle)


def build_joint_candidate_pool(
    candidate_pools: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind both role-specific five-family pools into one dataset claim."""

    if not isinstance(candidate_pools, Mapping) or tuple(candidate_pools) != JOINT_ROLE_ORDER:
        raise PBDRV4OfficialOnceError("joint candidate-pool role order differs")
    validated = {
        role: validate_candidate_pool(candidate_pools[role]) for role in JOINT_ROLE_ORDER
    }
    datasets = {pool.get("dataset") for pool in validated.values()}
    sources = {pool.get("source_lock_sha256") for pool in validated.values()}
    splits = {pool.get("split_projection_sha256") for pool in validated.values()}
    family_orders = {tuple(pool.get("family_order", ())) for pool in validated.values()}
    if len(datasets) != 1 or len(sources) != 1 or len(splits) != 1 or len(family_orders) != 1:
        raise PBDRV4OfficialOnceError("joint candidate pools do not share one context")
    for role in JOINT_ROLE_ORDER:
        if validated[role].get("role") != role:
            raise PBDRV4OfficialOnceError(f"joint candidate pool role differs: {role}")
    family_order = next(iter(family_orders))
    execution_keys = [
        f"{role}::{family}" for role in JOINT_ROLE_ORDER for family in family_order
    ]
    payload: dict[str, object] = {
        "schema": JOINT_CANDIDATE_POOL_SCHEMA,
        "status": "frozen_before_official_claim",
        "dataset": next(iter(datasets)),
        "role_order": list(JOINT_ROLE_ORDER),
        "family_order": list(family_order),
        "execution_keys": execution_keys,
        "candidate_count": len(execution_keys),
        "source_lock_sha256": next(iter(sources)),
        "split_projection_sha256": next(iter(splits)),
        "candidate_pool_sha256_by_role": {
            role: validated[role]["candidate_pool_sha256"] for role in JOINT_ROLE_ORDER
        },
        "candidate_pools": validated,
        "official_test_accessed": False,
    }
    payload["joint_candidate_pool_sha256"] = _canonical_sha(payload)
    return validate_joint_candidate_pool(payload)


def validate_joint_candidate_pool(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a two-role, ten-forward candidate plan without data access."""

    if (
        payload.get("schema") != JOINT_CANDIDATE_POOL_SCHEMA
        or payload.get("status") != "frozen_before_official_claim"
        or payload.get("official_test_accessed") is not False
    ):
        raise PBDRV4OfficialOnceError("joint candidate-pool identity/status differs")
    if payload.get("role_order") != list(JOINT_ROLE_ORDER):
        raise PBDRV4OfficialOnceError("joint candidate-pool role order differs")
    raw_pools = payload.get("candidate_pools")
    if not isinstance(raw_pools, Mapping) or tuple(raw_pools) != JOINT_ROLE_ORDER:
        raise PBDRV4OfficialOnceError("joint candidate-pool records differ")
    validated = {
        role: validate_candidate_pool(raw_pools[role]) for role in JOINT_ROLE_ORDER
    }
    datasets = {pool.get("dataset") for pool in validated.values()}
    sources = {pool.get("source_lock_sha256") for pool in validated.values()}
    splits = {pool.get("split_projection_sha256") for pool in validated.values()}
    family_orders = {tuple(pool.get("family_order", ())) for pool in validated.values()}
    if len(datasets) != 1 or len(sources) != 1 or len(splits) != 1 or len(family_orders) != 1:
        raise PBDRV4OfficialOnceError("joint candidate pools do not share one context")
    for role in JOINT_ROLE_ORDER:
        if validated[role].get("role") != role:
            raise PBDRV4OfficialOnceError(f"joint candidate pool role differs: {role}")
    family_order = next(iter(family_orders))
    execution_keys = [
        f"{role}::{family}" for role in JOINT_ROLE_ORDER for family in family_order
    ]
    expected = {
        "dataset": next(iter(datasets)),
        "family_order": list(family_order),
        "execution_keys": execution_keys,
        "candidate_count": len(execution_keys),
        "source_lock_sha256": next(iter(sources)),
        "split_projection_sha256": next(iter(splits)),
        "candidate_pool_sha256_by_role": {
            role: validated[role]["candidate_pool_sha256"] for role in JOINT_ROLE_ORDER
        },
    }
    for name, expected_value in expected.items():
        if payload.get(name) != expected_value:
            raise PBDRV4OfficialOnceError(f"joint candidate-pool {name} differs")
    declared = payload.get("joint_candidate_pool_sha256")
    unsigned = dict(payload)
    unsigned.pop("joint_candidate_pool_sha256", None)
    if declared != _canonical_sha(unsigned):
        raise PBDRV4OfficialOnceError("joint candidate-pool canonical SHA differs")
    return dict(payload)


def _validate_joint_bundle(
    bundle: Mapping[str, object],
    *,
    joint_candidate_pool: Mapping[str, object],
) -> dict[str, object]:
    if bundle.get("schema") != OFFICIAL_BUNDLE_SCHEMA or bundle.get("status") != "committed":
        raise PBDRV4OfficialOnceError("official joint bundle identity/status differs")
    if bundle.get("joint_candidate_pool_sha256") != joint_candidate_pool.get(
        "joint_candidate_pool_sha256"
    ):
        raise PBDRV4OfficialOnceError("official joint bundle candidate-pool binding differs")
    declared = bundle.get("bundle_sha256")
    unsigned = dict(bundle)
    unsigned.pop("bundle_sha256", None)
    if declared != _canonical_sha(unsigned):
        raise PBDRV4OfficialOnceError("official joint bundle canonical SHA differs")
    sample_count = bundle.get("sample_count")
    counts = bundle.get("forward_counts")
    execution_keys = joint_candidate_pool.get("execution_keys")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise PBDRV4OfficialOnceError("official joint bundle sample count differs")
    if not isinstance(counts, Mapping) or not isinstance(execution_keys, list):
        raise PBDRV4OfficialOnceError("official joint bundle forward counts differ")
    if set(counts) != set(execution_keys) or any(
        counts[key] != sample_count for key in execution_keys
    ):
        raise PBDRV4OfficialOnceError("official joint candidate forward count differs")
    if bundle.get("loader_iteration_count") != 1:
        raise PBDRV4OfficialOnceError("official joint loader iteration count differs")
    return dict(bundle)


def _validate_joint_preflight(
    preflight_result: Mapping[str, object],
    *,
    joint_candidate_pool: Mapping[str, object],
) -> dict[str, object]:
    result = dict(preflight_result)
    if result.get("official_test_accessed") is not False:
        raise PBDRV4OfficialOnceError("joint preflight crossed official boundary")
    expected = {
        "dataset": joint_candidate_pool["dataset"],
        "source_lock_sha256": joint_candidate_pool["source_lock_sha256"],
        "split_projection_sha256": joint_candidate_pool[
            "split_projection_sha256"
        ],
        "joint_candidate_pool_sha256": joint_candidate_pool[
            "joint_candidate_pool_sha256"
        ],
        "candidate_pool_sha256_by_role": joint_candidate_pool[
            "candidate_pool_sha256_by_role"
        ],
    }
    for name, expected_value in expected.items():
        if result.get(name) != expected_value:
            raise PBDRV4OfficialOnceError(
                f"joint preflight {name} binding differs"
            )
    return result


def execute_official_once(
    *,
    run_dir: Path,
    candidate_pool: Mapping[str, object],
    preflight: Callable[[], Mapping[str, object]],
    loader_factory: Callable[[], Iterable[Any]],
    consume_batch: Callable[[Any], Mapping[str, object]],
    finalize_metrics: Callable[[], Mapping[str, object]],
    materialize_views: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one claimed pass, or replay a committed bundle without data access.

    ``consume_batch`` returns ``sample_count`` and a ``forward_counts`` mapping
    for that batch.  It may update caller-owned online metric accumulators.
    No dataset or loader object is made before the exclusive claim is written.
    """

    pool = validate_candidate_pool(candidate_pool)
    root = Path(run_dir)
    if root.is_symlink():
        raise PBDRV4OfficialOnceError("official run directory is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    claim_path = root / "official_claim.json"
    failure_path = root / "consumed_failure.json"
    bundle_path = root / "publication_bundle.json"

    if bundle_path.exists() or bundle_path.is_symlink():
        bundle = _validate_bundle(
            _load_json(bundle_path, label="publication bundle"),
            candidate_pool=pool,
        )
        if materialize_views is not None:
            materialize_views(bundle)
        return bundle
    if claim_path.exists() or claim_path.is_symlink():
        state = "consumed_failure" if failure_path.exists() else "claimed_without_bundle"
        raise PBDRV4OfficialOnceError(
            f"official boundary already consumed ({state}); a second pass is forbidden"
        )

    preflight_result = dict(preflight())
    if preflight_result.get("official_test_accessed") is not False:
        raise PBDRV4OfficialOnceError("preflight crossed official boundary")
    claim: dict[str, object] = {
        "schema": OFFICIAL_CLAIM_SCHEMA,
        "status": "claimed",
        "dataset": pool["dataset"],
        "role": pool["role"],
        "candidate_pool_sha256": pool["candidate_pool_sha256"],
        "source_lock_sha256": pool["source_lock_sha256"],
        "preflight_sha256": _canonical_sha(preflight_result),
        "claim_time_ns": time.time_ns(),
        "dataset_or_loader_constructed_before_claim": False,
    }
    claim["claim_sha256"] = _canonical_sha(claim)
    _exclusive_json(claim_path, claim)

    total_samples = 0
    forward_counts = {family: 0 for family in pool["family_order"]}  # type: ignore[index]
    try:
        loader = loader_factory()
        iterator = iter(loader)
        loader_iteration_count = 1
        for batch in iterator:
            result = consume_batch(batch)
            batch_count = result.get("sample_count")
            batch_forwards = result.get("forward_counts")
            if (
                isinstance(batch_count, bool)
                or not isinstance(batch_count, int)
                or batch_count <= 0
                or not isinstance(batch_forwards, Mapping)
            ):
                raise PBDRV4OfficialOnceError("batch consumption counts are malformed")
            if set(batch_forwards) != set(forward_counts) or any(
                batch_forwards[name] != batch_count for name in forward_counts
            ):
                raise PBDRV4OfficialOnceError("one or more candidates were not forwarded once per sample")
            total_samples += batch_count
            for name in forward_counts:
                forward_counts[name] += int(batch_forwards[name])
        if total_samples <= 0 or any(value != total_samples for value in forward_counts.values()):
            raise PBDRV4OfficialOnceError("official pass is empty or forward counts differ")
        metrics = dict(finalize_metrics())
        bundle: dict[str, object] = {
            "schema": OFFICIAL_BUNDLE_SCHEMA,
            "status": "committed",
            "dataset": pool["dataset"],
            "role": pool["role"],
            "candidate_pool_sha256": pool["candidate_pool_sha256"],
            "source_lock_sha256": pool["source_lock_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "sample_count": total_samples,
            "forward_counts": forward_counts,
            "loader_iteration_count": loader_iteration_count,
            "metrics": metrics,
            "official_probability_or_logit_cache_written": False,
            "official_sweep_performed": False,
        }
        bundle["bundle_sha256"] = _canonical_sha(bundle)
        _validate_bundle(bundle, candidate_pool=pool)
        _exclusive_json(bundle_path, bundle)
    except BaseException as error:
        failure: dict[str, object] = {
            "schema": OFFICIAL_FAILURE_SCHEMA,
            "status": "consumed_terminal_failure",
            "candidate_pool_sha256": pool["candidate_pool_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "second_pass_forbidden": True,
        }
        failure["failure_sha256"] = _canonical_sha(failure)
        if not failure_path.exists() and not failure_path.is_symlink():
            _exclusive_json(failure_path, failure)
        raise

    committed = _validate_bundle(
        _load_json(bundle_path, label="publication bundle"),
        candidate_pool=pool,
    )
    if materialize_views is not None:
        materialize_views(committed)
    return committed


def execute_official_joint_once(
    *,
    run_dir: Path,
    joint_candidate_pool: Mapping[str, object],
    preflight: Callable[[], Mapping[str, object]],
    loader_factory: Callable[[], Iterable[Any]],
    consume_batch: Callable[[Any], Mapping[str, object]],
    finalize_metrics: Callable[[], Mapping[str, object]],
    materialize_views: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Consume both roles through one dataset-level claim and loader pass."""

    pool = validate_joint_candidate_pool(joint_candidate_pool)
    root = Path(run_dir)
    if root.is_symlink():
        raise PBDRV4OfficialOnceError("official run directory is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    claim_path = root / "official_claim.json"
    failure_path = root / "consumed_failure.json"
    bundle_path = root / "publication_bundle.json"

    if bundle_path.exists() or bundle_path.is_symlink():
        bundle = _validate_joint_bundle(
            _load_json(bundle_path, label="publication bundle"),
            joint_candidate_pool=pool,
        )
        if materialize_views is not None:
            materialize_views(bundle)
        return bundle
    if claim_path.exists() or claim_path.is_symlink():
        state = "consumed_failure" if failure_path.exists() else "claimed_without_bundle"
        raise PBDRV4OfficialOnceError(
            f"official boundary already consumed ({state}); a second pass is forbidden"
        )

    preflight_result = _validate_joint_preflight(
        preflight(),
        joint_candidate_pool=pool,
    )
    claim: dict[str, object] = {
        "schema": OFFICIAL_CLAIM_SCHEMA,
        "status": "claimed",
        "dataset": pool["dataset"],
        "role_order": pool["role_order"],
        "joint_candidate_pool_sha256": pool["joint_candidate_pool_sha256"],
        "candidate_pool_sha256_by_role": pool["candidate_pool_sha256_by_role"],
        "source_lock_sha256": pool["source_lock_sha256"],
        "preflight_sha256": _canonical_sha(preflight_result),
        "claim_time_ns": time.time_ns(),
        "dataset_or_loader_constructed_before_claim": False,
    }
    claim["claim_sha256"] = _canonical_sha(claim)
    _exclusive_json(claim_path, claim)

    total_samples = 0
    execution_keys = list(pool["execution_keys"])  # type: ignore[arg-type]
    forward_counts = {key: 0 for key in execution_keys}
    try:
        loader = loader_factory()
        iterator = iter(loader)
        loader_iteration_count = 1
        for batch in iterator:
            result = consume_batch(batch)
            batch_count = result.get("sample_count")
            batch_forwards = result.get("forward_counts")
            if (
                isinstance(batch_count, bool)
                or not isinstance(batch_count, int)
                or batch_count <= 0
                or not isinstance(batch_forwards, Mapping)
            ):
                raise PBDRV4OfficialOnceError("joint batch consumption counts are malformed")
            if set(batch_forwards) != set(forward_counts) or any(
                batch_forwards[name] != batch_count for name in forward_counts
            ):
                raise PBDRV4OfficialOnceError(
                    "one or more joint candidates were not forwarded once per sample"
                )
            total_samples += batch_count
            for name in forward_counts:
                forward_counts[name] += int(batch_forwards[name])
        if total_samples <= 0 or any(value != total_samples for value in forward_counts.values()):
            raise PBDRV4OfficialOnceError(
                "official joint pass is empty or forward counts differ"
            )
        metrics = dict(finalize_metrics())
        bundle: dict[str, object] = {
            "schema": OFFICIAL_BUNDLE_SCHEMA,
            "status": "committed",
            "dataset": pool["dataset"],
            "role_order": pool["role_order"],
            "joint_candidate_pool_sha256": pool["joint_candidate_pool_sha256"],
            "candidate_pool_sha256_by_role": pool["candidate_pool_sha256_by_role"],
            "source_lock_sha256": pool["source_lock_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "sample_count": total_samples,
            "forward_counts": forward_counts,
            "loader_iteration_count": loader_iteration_count,
            "metrics": metrics,
            "official_probability_or_logit_cache_written": False,
            "official_sweep_performed": False,
        }
        bundle["bundle_sha256"] = _canonical_sha(bundle)
        _validate_joint_bundle(bundle, joint_candidate_pool=pool)
        _exclusive_json(bundle_path, bundle)
    except BaseException as error:
        failure: dict[str, object] = {
            "schema": OFFICIAL_FAILURE_SCHEMA,
            "status": "consumed_terminal_failure",
            "joint_candidate_pool_sha256": pool["joint_candidate_pool_sha256"],
            "claim_sha256": claim["claim_sha256"],
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "second_pass_forbidden": True,
        }
        failure["failure_sha256"] = _canonical_sha(failure)
        if not failure_path.exists() and not failure_path.is_symlink():
            _exclusive_json(failure_path, failure)
        raise

    committed = _validate_joint_bundle(
        _load_json(bundle_path, label="publication bundle"),
        joint_candidate_pool=pool,
    )
    if materialize_views is not None:
        materialize_views(committed)
    return committed


__all__ = [
    "JOINT_CANDIDATE_POOL_SCHEMA",
    "JOINT_ROLE_ORDER",
    "OFFICIAL_BUNDLE_SCHEMA",
    "OFFICIAL_CLAIM_SCHEMA",
    "OFFICIAL_FAILURE_SCHEMA",
    "PBDRV4OfficialOnceError",
    "build_joint_candidate_pool",
    "execute_official_joint_once",
    "execute_official_once",
    "validate_joint_candidate_pool",
]
