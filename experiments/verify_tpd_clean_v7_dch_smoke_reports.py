#!/usr/bin/env python3
"""Verify the source-bound CPU/GPU2/GPU3 V7-DCH smoke-report set."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    capture_tpd_clean_v7_dch_smoke_report as capture,
)
from experiments.smoke_tpd_clean_v7_dch import (  # noqa: E402
    EXPECTED_BLOCK_EPS_NAMES,
    EXPECTED_DEVICE_NAME,
    EXPECTED_KEEP_PARAMETER_NAMES,
    EXPECTED_SCALE_PARAMETER_NAMES,
    FORMAL_EPS,
    PHYSICAL_GPU_UUIDS,
    SCHEMA as SMOKE_SCHEMA,
)
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    CONTEXT_CODE_FORMULA,
    FULL_HEADROOM_FORMULA,
    FUSION_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
)
from model.tpd_clean_v7_dch import (  # noqa: E402
    CONTEXT_HEADROOM_CEILING,
    CONTEXT_HEADROOM_FLOOR,
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
)


SCHEMA = "sctransnet_tpd_clean_v7_dch_smoke_verification_v1"
PRIMARY_VARIANT = "tpd_clean_v7_dch_full"
CONTROL_VARIANT = "tpd_clean_v7_dch_capacity"
EXPECTED_REPORTS: Mapping[str, Mapping[str, Any]] = {
    "cpu_all.json": {
        "device": "cpu",
        "device_name": "cpu",
        "physical_index": None,
        "device_uuid": None,
        "batch_size": 2,
        "patch_size": 32,
        "variants": list(SUPPORTED_CLEAN_V7_DCH_VARIANTS),
        "paired": True,
        "paired_status": "verified",
        "first_step_paired": True,
    },
    "gpu2_full.json": {
        "device": "cuda:0",
        "device_name": EXPECTED_DEVICE_NAME,
        "physical_index": "2",
        "device_uuid": PHYSICAL_GPU_UUIDS["2"],
        "batch_size": 2,
        "patch_size": 64,
        "variants": [PRIMARY_VARIANT],
        "paired": None,
        "paired_status": "not_checked_single_variant",
        "first_step_paired": None,
    },
    "gpu3_capacity.json": {
        "device": "cuda:0",
        "device_name": EXPECTED_DEVICE_NAME,
        "physical_index": "3",
        "device_uuid": PHYSICAL_GPU_UUIDS["3"],
        "batch_size": 2,
        "patch_size": 64,
        "variants": [CONTROL_VARIANT],
        "paired": None,
        "paired_status": "not_checked_single_variant",
        "first_step_paired": None,
    },
}


class SmokeReportError(RuntimeError):
    """Raised when a persisted DCH smoke report violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeReportError(message)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label}: SHA256",
    )
    return value


def _positive_mapping(
    value: Any,
    expected_names: frozenset[str],
    label: str,
) -> None:
    _require(isinstance(value, dict), f"{label}: expected mapping")
    _require(set(value) == set(expected_names), f"{label}: names differ")
    _require(
        all(_is_number(item) and float(item) > 0.0 for item in value.values()),
        f"{label}: values must be finite and positive",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label}: missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeReportError(f"{label}: invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label}: expected object")
    return payload


def _validate_timestamp(value: Any, label: str) -> None:
    _require(isinstance(value, str), f"{label}: timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise SmokeReportError(f"{label}: timestamp") from exc
    _require(parsed.tzinfo is not None, f"{label}: timezone")


def _validate_device(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    contract = report.get("cuda_device_contract")
    _require(isinstance(contract, dict), f"{label}: device contract")
    physical_index = expected["physical_index"]
    if physical_index is None:
        _require(report.get("cuda_visible_devices") is None, f"{label}: mask")
        _require(contract.get("applicable") is False, f"{label}: CPU")
        _require(contract.get("validated") is False, f"{label}: CPU")
        return
    expected_uuid = expected["device_uuid"]
    _require(
        report.get("environment_cuda_visible_devices") == physical_index
        and report.get("cuda_visible_devices") == physical_index,
        f"{label}: physical mask",
    )
    _require(
        contract.get("applicable") is True
        and contract.get("validated") is True,
        f"{label}: CUDA validation",
    )
    _require(
        contract.get("declared_physical_index") == physical_index
        and contract.get("expected_physical_index") == physical_index,
        f"{label}: physical index",
    )
    _require(
        contract.get("logical_device") == "cuda:0"
        and contract.get("visible_device_count") == 1,
        f"{label}: logical device",
    )
    _require(
        contract.get("device_name") == EXPECTED_DEVICE_NAME
        and contract.get("expected_device_name") == EXPECTED_DEVICE_NAME,
        f"{label}: device name",
    )
    _require(
        contract.get("device_uuid") == expected_uuid
        and contract.get("expected_device_uuid") == expected_uuid,
        f"{label}: device UUID",
    )


def _validate_variant(
    entry: Mapping[str, Any],
    expected_variant: str,
    label: str,
) -> str:
    _require(entry.get("variant") == expected_variant, f"{label}: variant")
    _require(entry.get("status") == "complete", f"{label}: status")
    _require(entry.get("output_count") == 6, f"{label}: outputs")
    _require(entry.get("loss_head_count") == 6, f"{label}: loss heads")
    _require(
        entry.get("loss_definition") == "sum_of_six_mean_bce_outputs"
        and entry.get("loss_sum_verified") is True,
        f"{label}: loss",
    )
    losses = entry.get("losses")
    per_head = entry.get("per_head_losses")
    _require(
        isinstance(losses, list)
        and len(losses) == 2
        and all(_is_number(value) for value in losses),
        f"{label}: total losses",
    )
    _require(
        isinstance(per_head, list)
        and len(per_head) == 2
        and all(
            isinstance(row, list)
            and len(row) == 6
            and all(_is_number(value) for value in row)
            for row in per_head
        ),
        f"{label}: head losses",
    )
    _require(
        entry.get("optimizer_steps_completed") == 2,
        f"{label}: steps",
    )
    _require(
        entry.get("step_zero_exact_spd") is True
        and entry.get("step_zero_max_abs_difference") == 0.0,
        f"{label}: zero-scale SPD",
    )
    _sha256(
        entry.get("first_step_model_checksum"),
        f"{label}: first-step model",
    )
    _sha256(
        entry.get("first_step_optimizer_checksum"),
        f"{label}: first-step optimizer",
    )
    optimizer_state = entry.get("first_step_optimizer_state")
    _require(
        isinstance(optimizer_state, dict)
        and int(optimizer_state.get("parameter_state_count", 0)) > 0
        and optimizer_state.get("step_values") == [1.0]
        and _is_number(optimizer_state.get("exp_avg_l1"))
        and float(optimizer_state["exp_avg_l1"]) > 0.0
        and _is_number(optimizer_state.get("exp_avg_sq_l1"))
        and float(optimizer_state["exp_avg_sq_l1"]) > 0.0,
        f"{label}: first-step Adam state",
    )
    _positive_mapping(
        entry.get("scale_gradient_l1"),
        EXPECTED_SCALE_PARAMETER_NAMES,
        f"{label}: scale gradients",
    )
    _positive_mapping(
        entry.get("phase_gradient_l1"),
        EXPECTED_KEEP_PARAMETER_NAMES,
        f"{label}: Keep gradients",
    )
    _positive_mapping(
        entry.get("scale_update_l1"),
        EXPECTED_SCALE_PARAMETER_NAMES,
        f"{label}: scale updates",
    )
    _positive_mapping(
        entry.get("phase_update_l1"),
        EXPECTED_KEEP_PARAMETER_NAMES,
        f"{label}: Keep updates",
    )
    _require(
        entry.get("strict_rebuild_load") is True
        and entry.get("strict_reload_max_abs_difference") == 0.0,
        f"{label}: strict reload",
    )
    _require(entry.get("total_parameters") == 10_843_155, f"{label}: params")
    _require(
        entry.get("shallow_embedding_parameters") == 66_176,
        f"{label}: shallow params",
    )
    _require(
        entry.get("phase_tied_projection")
        == "sum_keep_weights_over_four_contiguous_phases"
        and entry.get("derived_projection_parameters") == 0,
        f"{label}: phase-tied projection",
    )
    block_eps = entry.get("block_eps")
    _require(
        isinstance(block_eps, dict)
        and set(block_eps) == set(EXPECTED_BLOCK_EPS_NAMES)
        and all(value == FORMAL_EPS for value in block_eps.values()),
        f"{label}: block eps",
    )
    _require(
        entry.get("amp_enabled") is False
        and entry.get("autocast_forced_disabled") is True,
        f"{label}: FP32",
    )
    _require(
        entry.get("input_dtype") == "torch.float32"
        and entry.get("target_dtype") == "torch.float32"
        and entry.get("output_dtypes") == ["torch.float32"]
        and entry.get("model_parameter_dtypes") == ["torch.float32"],
        f"{label}: dtype",
    )
    _require(
        entry.get("model_floating_buffer_dtypes")
        in ([], ["torch.float32"]),
        f"{label}: buffer dtype",
    )
    for precision_name in (
        "projection_precision",
        "context_precision",
        "coefficient_precision",
    ):
        _require(
            entry.get(precision_name)
            == "float32_in_formal_amp_off_path",
            f"{label}: {precision_name}",
        )
    _require(
        entry.get("residual_output_dtype") == "feature_dtype",
        f"{label}: residual dtype",
    )
    if expected_variant == PRIMARY_VARIANT:
        _require(entry.get("context_gate") == 1.0, f"{label}: Full gate")
        _require(
            entry.get("context_modulation") == "half_centered_context_code",
            f"{label}: Full context",
        )
    else:
        _require(entry.get("context_gate") == 0.0, f"{label}: control gate")
        _require(
            entry.get("context_code")
            == "not_computed_in_capacity_forward",
            f"{label}: Capacity context",
        )
    _sha256(
        entry.get("trained_model_checksum"),
        f"{label}: trained model",
    )
    return _sha256(
        entry.get("initial_model_checksum"),
        f"{label}: initial model",
    )


def _validate_report(
    report: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> set[str]:
    _require(
        report.get("schema") == SMOKE_SCHEMA
        and report.get("status") == "complete",
        f"{label}: schema/status",
    )
    _require(
        report.get("device") == expected["device"]
        and report.get("device_name") == expected["device_name"],
        f"{label}: device",
    )
    _require(
        report.get("batch_size") == expected["batch_size"]
        and report.get("patch_size") == expected["patch_size"]
        and report.get("steps") == 2
        and report.get("seed") == 42,
        f"{label}: smoke axes",
    )
    _require(
        report.get("formal_eps") == FORMAL_EPS
        and report.get("formal_amp_enabled") is False
        and report.get("autocast_forced_disabled") is True,
        f"{label}: formal FP32",
    )
    _require(
        report.get("input_dtype") == "torch.float32"
        and report.get("target_dtype") == "torch.float32",
        f"{label}: formal input dtype",
    )
    _require(
        report.get("scale_parameter_names")
        == sorted(EXPECTED_SCALE_PARAMETER_NAMES)
        and report.get("keep_parameter_names")
        == sorted(EXPECTED_KEEP_PARAMETER_NAMES),
        f"{label}: parameter names",
    )
    _require(
        report.get("phase_tied_projection_formula")
        == PHASE_TIED_PROJECTION_FORMULA
        and report.get("context_code_formula") == CONTEXT_CODE_FORMULA
        and report.get("deferred_context_headroom_formula")
        == FULL_HEADROOM_FORMULA
        and report.get("fusion_equation") == FUSION_FORMULA,
        f"{label}: formulas",
    )
    _require(
        report.get("headroom_bound")
        == [CONTEXT_HEADROOM_FLOOR, CONTEXT_HEADROOM_CEILING],
        f"{label}: headroom",
    )
    _require(
        report.get("learned_scales_per_block") == 1
        and report.get("derived_projection_parameters") == 0
        and report.get("coefficient_bound") == "abs(a*H)<=1"
        and report.get("residual_bound") == "abs(R)<=abs(Sa)",
        f"{label}: DCH residual contract",
    )
    _require(
        report.get("paired_initialization") is expected["paired"]
        and report.get("paired_initialization_status")
        == expected["paired_status"]
        and report.get("paired_first_adam_step_exact")
        is expected["first_step_paired"],
        f"{label}: pairing",
    )
    if expected["paired"] is True:
        _sha256(
            report.get("paired_initialization_sha256"),
            f"{label}: paired initialization",
        )
    else:
        _require(
            report.get("paired_initialization_sha256") is None,
            f"{label}: single pairing",
        )
    _validate_device(report, expected, label)
    variants = report.get("variants")
    _require(isinstance(variants, list), f"{label}: variants")
    _require(
        [entry.get("variant") for entry in variants] == expected["variants"],
        f"{label}: variant order",
    )
    initial = {
        _validate_variant(
            entry,
            expected_variant,
            f"{label}/{expected_variant}",
        )
        for entry, expected_variant in zip(variants, expected["variants"])
    }
    if expected["paired"] is True:
        _require(len(initial) == 1, f"{label}: initialization equality")
        _require(
            report.get("paired_initialization_sha256") == next(iter(initial)),
            f"{label}: initialization record",
        )
    cuda_memory = report.get("cuda_memory")
    if expected["device"] == "cpu":
        _require(cuda_memory is None, f"{label}: CPU memory")
    else:
        _require(
            isinstance(cuda_memory, dict)
            and set(cuda_memory)
            == {"peak_allocated_mib", "peak_reserved_mib"}
            and all(
                _is_number(value) and float(value) > 0.0
                for value in cuda_memory.values()
            ),
            f"{label}: GPU memory",
        )
    return initial


def validate_smoke_reports(smoke_root: Path) -> dict[str, Any]:
    smoke_root = smoke_root.resolve()
    _require(smoke_root.is_dir(), "DCH smoke root is not a directory")
    observed = {
        path.name for path in smoke_root.iterdir() if path.suffix == ".json"
    }
    _require(
        observed == set(EXPECTED_REPORTS),
        "DCH smoke root must contain exactly cpu_all.json, "
        "gpu2_full.json, and gpu3_capacity.json",
    )
    current_sources = capture.source_manifest()
    initial_checksums: set[str] = set()
    report_sha256: dict[str, str] = {}
    for name, expected in EXPECTED_REPORTS.items():
        path = smoke_root / name
        envelope = _load_json(path, f"smoke/{name}")
        _require(
            envelope.get("schema") == capture.SCHEMA
            and envelope.get("status") == "complete",
            f"smoke/{name}: envelope",
        )
        _validate_timestamp(
            envelope.get("created_at_utc"),
            f"smoke/{name}",
        )
        _require(
            envelope.get("source_sha256") == current_sources,
            f"smoke/{name}: source manifest",
        )
        report = envelope.get("report")
        _require(isinstance(report, dict), f"smoke/{name}: report")
        _require(
            envelope.get("environment_cuda_visible_devices")
            == report.get("environment_cuda_visible_devices")
            and envelope.get("cuda_visible_devices")
            == report.get("cuda_visible_devices")
            and envelope.get("cuda_device_contract")
            == report.get("cuda_device_contract"),
            f"smoke/{name}: device binding",
        )
        initial_checksums.update(
            _validate_report(report, expected, f"smoke/{name}")
        )
        report_sha256[name] = capture.file_sha256(path)
    _require(
        len(initial_checksums) == 1,
        "CPU/GPU2/GPU3 DCH reports do not share one initialization",
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "smoke_root": str(smoke_root),
        "expected_report_names": sorted(EXPECTED_REPORTS),
        "report_sha256": report_sha256,
        "source_sha256": current_sources,
        "cross_report_initialization_verified": True,
        "paired_initialization_sha256": next(iter(initial_checksums)),
        "physical_gpu_reports_verified": {
            "2": PHYSICAL_GPU_UUIDS["2"],
            "3": PHYSICAL_GPU_UUIDS["3"],
        },
        "passed": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the V7-DCH CPU/GPU2/GPU3 smoke reports"
    )
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = validate_smoke_reports(args.smoke_root)
    if args.output is None:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        return
    output = (
        args.output if args.output.is_absolute() else REPO_ROOT / args.output
    )
    capture.exclusive_write_json(output, result)
    print(
        f"TPDCLEANV7DCH_SMOKE_REPORT_SET_OK output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
