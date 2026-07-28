#!/usr/bin/env python3
"""Create or verify the separate V8 training and acceptance source locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRAINING_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_exact_source_lock_v1"
)
ACCEPTANCE_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_acceptance_source_lock_v1"
)
CANDIDATE_FAMILY = "tpd_clean_v8_mprs_dch"
DATASET = "NUDT-SIRST"
VARIANTS = (
    "tpd_clean_v8_mprs_dch_full",
    "tpd_clean_v8_mprs_dch_capacity",
)
DEFAULT_TRAINING_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v8_mprs_dch_exact_source_lock.json"
)
DEFAULT_ACCEPTANCE_LOCK = (
    REPO_ROOT
    / "experiments/tpd_clean_v8_mprs_dch_acceptance_source_lock.json"
)
ACCEPTANCE_SOURCE_RELATIVES = (
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "analysis/analyze_tpd_clean_v8_mprs_mechanism.py",
    "analysis/benchmark_tpd_clean_v8_mprs_dch.py",
    "experiments/smoke_tpd_clean_v8_mprs_dch.py",
    "experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh",
    "experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_repo_file(repo_root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("source relative path must be a non-empty string")
    root = repo_root.resolve()
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {relative}")
    resolved = candidate.resolve()
    try:
        canonical = str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"source escapes repository: {relative}") from exc
    if canonical != relative:
        raise ValueError(
            f"source path is not canonical: supplied={relative!r}, "
            f"canonical={canonical!r}"
        )
    return resolved


def hash_sources(
    repo_root: Path,
    relatives: Iterable[str],
) -> dict[str, str]:
    ordered = tuple(relatives)
    if len(ordered) != len(set(ordered)):
        raise ValueError("source list contains duplicate paths")
    return {
        relative: file_sha256(_regular_repo_file(repo_root, relative))
        for relative in ordered
    }


def training_source_relatives() -> tuple[str, ...]:
    from experiments import train_tpd_clean_v8_mprs_dch_exact as exact

    root = exact.REPO_ROOT.resolve()
    relatives = tuple(
        str(path.resolve().relative_to(root))
        for path in exact.RUNTIME_SOURCE_PATHS
    )
    if len(relatives) != len(set(relatives)):
        raise RuntimeError("V8 exact runtime source list contains duplicates")
    return relatives


def formal_contract() -> dict[str, Any]:
    from experiments import train_tpd_clean_v8_mprs_dch_exact as exact

    return dict(exact.formal_contract())


def _field_update(
    digest: "hashlib._Hash",
    label: str,
    payload: bytes,
) -> None:
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _file_update(
    digest: "hashlib._Hash",
    label: str,
    path: Path,
) -> None:
    encoded = label.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(path.stat().st_size.to_bytes(8, "big"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def _resolve_unique_sample_file(directory: Path, identifier: str) -> Path:
    extensions = (".png", ".bmp", ".jpg", ".jpeg")
    matches = [
        directory / f"{identifier}{extension}"
        for extension in extensions
        if (directory / f"{identifier}{extension}").is_file()
    ]
    if len(matches) != 1 or matches[0].is_symlink():
        raise ValueError(
            f"expected one regular sample file for {identifier!r} "
            f"under {directory}, found {len(matches)}"
        )
    return matches[0]


def training_data_contract(
    dataset_dir: Path,
    dataset: str = DATASET,
) -> dict[str, Any]:
    dataset_root = dataset_dir.resolve() / dataset
    index_relative = f"img_idx/train_{dataset}.txt"
    index_path = dataset_root / index_relative
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError(f"official training index is missing: {index_path}")
    index_bytes = index_path.read_bytes()
    try:
        identifiers = [
            line.strip()
            for line in index_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except UnicodeDecodeError as exc:
        raise ValueError("official training index is not UTF-8") from exc
    if len(identifiers) < 2 or len(identifiers) != len(set(identifiers)):
        raise ValueError(
            "official training index must contain at least two unique IDs"
        )
    digest = hashlib.sha256()
    _field_update(digest, "schema", b"tpd-training-data-fingerprint-v1")
    _field_update(digest, index_relative, index_bytes)
    for identifier in identifiers:
        for directory_name in ("images", "masks"):
            sample = _resolve_unique_sample_file(
                dataset_root / directory_name,
                identifier,
            )
            _file_update(
                digest,
                f"{directory_name}/{sample.name}",
                sample,
            )
    ordered_ids_payload = json.dumps(
        identifiers,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "dataset": dataset,
        "official_training_index": index_relative,
        "official_training_sample_count": len(identifiers),
        "official_training_index_sha256": hashlib.sha256(
            index_bytes
        ).hexdigest(),
        "ordered_training_ids_sha256": hashlib.sha256(
            ordered_ids_payload
        ).hexdigest(),
        "training_data_sha256": digest.hexdigest(),
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
    data = training_data_contract(dataset_dir)
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
            "fresh_or_exact_resume_only": True,
            "cross_version_exact_resume_forbidden": True,
            "existing_lock_overwrite_forbidden": True,
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
        raise ValueError("training source-lock schema mismatch")
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
            "performance_metrics": ["Pd", "Fa", "mIoU"],
            "gates_A_to_E_unchanged": True,
            "checkpoint_reselection_permitted": False,
            "existing_lock_overwrite_forbidden": True,
            "source_symlinks_forbidden": True,
        },
    }


def load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"lock must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON lock: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"lock must contain one JSON object: {path}")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def publish_new_lock(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or resolved.exists():
        raise FileExistsError(f"source lock already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary lock path exists: {temporary}")
    try:
        temporary.write_text(_canonical_json(payload), encoding="utf-8")
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_source_mapping(
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> None:
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source lock has no source_sha256 mapping")
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
        raise ValueError("training source-lock identity differs")
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
        else dataset_dir
    )
    for field, expected in live_data.items():
        if payload.get(field) != expected:
            raise ValueError(f"training data contract differs: {field}")
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
        raise ValueError("acceptance source-lock identity differs")
    training = load_json_object(training_lock_path)
    if training.get("schema") != TRAINING_SCHEMA:
        raise ValueError("bound training source-lock schema differs")
    if (
        payload.get("training_source_lock_sha256")
        != file_sha256(training_lock_path)
        or payload.get("training_data_sha256")
        != training.get("training_data_sha256")
    ):
        raise ValueError("acceptance lock training binding differs")
    if set(payload.get("source_sha256", ())) != set(
        expected_source_relatives
    ):
        raise ValueError("acceptance source set differs")
    _verify_source_mapping(payload, repo_root=repo_root)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify V8-MPRS-DCH source locks"
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
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--training-lock", type=Path, default=DEFAULT_TRAINING_LOCK)
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
            training = build_training_lock(dataset_dir=args.dataset_dir)
            publish_new_lock(args.training_lock, training)
        if args.kind in ("acceptance", "all"):
            acceptance = build_acceptance_lock(args.training_lock)
            publish_new_lock(args.acceptance_lock, acceptance)
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
    outputs = {}
    if args.training_lock.is_file():
        outputs["training_source_lock_sha256"] = file_sha256(
            args.training_lock
        )
    if args.acceptance_lock.is_file():
        outputs["acceptance_source_lock_sha256"] = file_sha256(
            args.acceptance_lock
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "mode": args.mode,
                "kind": args.kind,
                **outputs,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
