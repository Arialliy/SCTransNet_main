#!/usr/bin/env python3
"""Build the immutable preflight pairing manifest for V7-DCH formal800."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import capture_tpd_clean_v7_dch_smoke_report as capture  # noqa: E402
from experiments import freeze_tpd_clean_v7_dch_source_locks as locks  # noqa: E402
from experiments import verify_tpd_clean_v7_dch_smoke_reports as smoke  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v7_dch_pairing_manifest_v1"
VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
SEEDS = (42, 3407)
GPU_UUIDS = {
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
SCHEDULE = (
    {
        "lane": "gpu2",
        "order": 1,
        "physical_gpu_index": 2,
        "physical_gpu_uuid": GPU_UUIDS[2],
        "logical_device": "cuda:0",
        "variant": "tpd_clean_v7_dch_full",
        "seed": 42,
    },
    {
        "lane": "gpu2",
        "order": 2,
        "physical_gpu_index": 2,
        "physical_gpu_uuid": GPU_UUIDS[2],
        "logical_device": "cuda:0",
        "variant": "tpd_clean_v7_dch_capacity",
        "seed": 3407,
    },
    {
        "lane": "gpu3",
        "order": 1,
        "physical_gpu_index": 3,
        "physical_gpu_uuid": GPU_UUIDS[3],
        "logical_device": "cuda:0",
        "variant": "tpd_clean_v7_dch_capacity",
        "seed": 42,
    },
    {
        "lane": "gpu3",
        "order": 2,
        "physical_gpu_index": 3,
        "physical_gpu_uuid": GPU_UUIDS[3],
        "logical_device": "cuda:0",
        "variant": "tpd_clean_v7_dch_full",
        "seed": 3407,
    },
)
DEFAULT_PREFLIGHT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v7_dch_preflight_v1"
)
DEFAULT_TRAINING_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v7_dch_exact_source_lock.json"
)
PAIR_REPORTS = {
    42: DEFAULT_PREFLIGHT_ROOT / "smoke_reports/cpu_all.json",
    3407: DEFAULT_PREFLIGHT_ROOT / "pairing_reports/cpu_all_seed3407.json",
}
EVIDENCE_SOURCES = (
    REPO_ROOT / "model/tpd_clean_v7_dch.py",
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch.py",
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch_exact.py",
    REPO_ROOT / "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md",
    REPO_ROOT / "tests/test_tpd_clean_v7_dch.py",
    Path(__file__).resolve(),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _pair_evidence(path: Path, seed: int) -> dict[str, Any]:
    envelope = _load_json(path, f"seed {seed} pairing report")
    _require(envelope.get("schema") == capture.SCHEMA, "pair report schema")
    _require(envelope.get("status") == "complete", "pair report status")
    _require(
        envelope.get("source_sha256") == capture.source_manifest(),
        "pair report source manifest differs from current code",
    )
    report = envelope.get("report")
    _require(isinstance(report, dict), "pair report payload")
    _require(report.get("status") == "complete", "pair smoke status")
    _require(report.get("device") == "cpu", "pair smoke must use CPU")
    _require(report.get("seed") == seed, "pair smoke seed")
    _require(report.get("paired_initialization") is True, "paired init")
    _require(
        report.get("paired_initialization_status") == "verified",
        "paired init status",
    )
    _require(
        report.get("paired_first_adam_step_exact") is True,
        "paired first Adam step",
    )

    variants = report.get("variants")
    _require(isinstance(variants, list) and len(variants) == 2, "pair variants")
    by_variant = {
        item.get("variant"): item
        for item in variants
        if isinstance(item, dict)
    }
    _require(tuple(by_variant) == VARIANTS, "pair variant order/identity")
    full, capacity = (by_variant[name] for name in VARIANTS)
    exact_equal_fields = (
        "initial_model_checksum",
        "first_step_model_checksum",
        "first_step_optimizer_checksum",
    )
    for field in exact_equal_fields:
        _require(full.get(field) == capacity.get(field), f"pair {field}")
    for candidate in (full, capacity):
        _require(candidate.get("status") == "complete", "candidate status")
        _require(candidate.get("total_parameters") == 10_843_155, "params")
        _require(
            candidate.get("shallow_embedding_parameters") == 66_176,
            "shallow params",
        )
        _require(candidate.get("output_count") == 6, "six outputs")
        _require(candidate.get("loss_sum_verified") is True, "six BCE loss")
        _require(candidate.get("step_zero_exact_spd") is True, "SPD anchor")
        _require(
            candidate.get("step_zero_max_abs_difference") == 0.0,
            "zero-scale output difference",
        )
        _require(
            candidate.get("strict_rebuild_load") is True,
            "strict rebuild/load",
        )
        _require(
            candidate.get("strict_reload_max_abs_difference") == 0.0,
            "strict reload output",
        )
        _require(
            candidate.get("optimizer_steps_completed") == 2,
            "optimizer steps",
        )
    return {
        "report": str(Path(path).resolve().relative_to(REPO_ROOT)),
        "report_sha256": file_sha256(path),
        "seed": seed,
        "paired_initialization_sha256": report[
            "paired_initialization_sha256"
        ],
        "initial_model_checksum": full["initial_model_checksum"],
        "first_step_model_checksum": full["first_step_model_checksum"],
        "first_step_optimizer_checksum": full[
            "first_step_optimizer_checksum"
        ],
        "zero_scale_outputs_exact": True,
        "zero_scale_loss_and_gradient_contract": (
            "bound_to_tests/test_tpd_clean_v7_dch.py"
        ),
        "first_adam_step_model_and_optimizer_exact": True,
    }


def build_pairing_manifest(
    *,
    preflight_root: Path = DEFAULT_PREFLIGHT_ROOT,
    training_lock: Path = DEFAULT_TRAINING_LOCK,
    pair_reports: Mapping[int, Path] | None = None,
) -> dict[str, Any]:
    """Validate all preflight evidence and return one pairing manifest."""

    preflight_root = Path(preflight_root).resolve()
    reports = (
        {seed: Path(path) for seed, path in pair_reports.items()}
        if pair_reports is not None
        else {
            42: preflight_root / "smoke_reports/cpu_all.json",
            3407: preflight_root / "pairing_reports/cpu_all_seed3407.json",
        }
    )
    _require(tuple(sorted(reports)) == SEEDS, "pair reports must cover both seeds")

    training_payload, training_lock_sha256 = locks.validate_source_lock(
        "training",
        Path(training_lock),
    )
    smoke_result = smoke.validate_smoke_reports(
        preflight_root / "smoke_reports"
    )
    _require(smoke_result.get("passed") is True, "persistent smoke set")

    evidence = [_pair_evidence(reports[seed], seed) for seed in SEEDS]
    source_sha256 = {
        str(path.relative_to(REPO_ROOT)): file_sha256(path)
        for path in EVIDENCE_SOURCES
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "passed": True,
        "mainline_contract": "Keep-Context-Saliency",
        "mainline_changed": False,
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "formal_runs": 4,
        "schedule": [dict(item) for item in SCHEDULE],
        "pair_evidence": evidence,
        "training_source_lock": {
            "path": str(Path(training_lock).resolve().relative_to(REPO_ROOT)),
            "sha256": training_lock_sha256,
            "source_count": training_payload["source_count"],
            "training_data_sha256": training_payload[
                "training_data_sha256"
            ],
        },
        "persistent_smoke": {
            "passed": True,
            "physical_gpu_reports_verified": smoke_result[
                "physical_gpu_reports_verified"
            ],
            "report_sha256": smoke_result["report_sha256"],
        },
        "source_sha256": source_sha256,
        "zero_scale_contract": {
            "outputs_loss_input_grad_shared_param_grads_exact": True,
            "first_adam_model_optimizer_state_exact": True,
            "test_source": "tests/test_tpd_clean_v7_dch.py",
        },
        "dch_causal_mechanism_established": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace pairing manifest: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(f"invalid pairing manifest parent: {path.parent}")
    content = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and persist the V7-DCH paired-start manifest"
    )
    parser.add_argument(
        "--preflight-root",
        type=Path,
        default=DEFAULT_PREFLIGHT_ROOT,
    )
    parser.add_argument(
        "--training-lock",
        type=Path,
        default=DEFAULT_TRAINING_LOCK,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREFLIGHT_ROOT / "pairing_manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_pairing_manifest(
        preflight_root=args.preflight_root,
        training_lock=args.training_lock,
    )
    output = write_new_json(args.output, payload)
    print(
        f"TPDCLEANV7DCH_PAIRING_OK output={output} "
        f"runs={payload['formal_runs']} seeds={len(payload['seeds'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
