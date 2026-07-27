from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import capture_tpd_clean_v6_smoke_report as capture
from experiments import verify_tpd_clean_v6_smoke_reports as verifier
from experiments.smoke_tpd_clean_v6 import (
    EXPECTED_DEVICE_NAME,
    EXPECTED_KEEP_PARAMETER_NAMES,
    EXPECTED_SCALE_PARAMETER_NAMES,
    FORMAL_EPS,
    PHYSICAL_GPU_UUIDS,
    SCHEMA as SMOKE_SCHEMA,
)
from experiments.train_tpd_clean_v6 import (
    CONTEXT_CODE_FORMULA,
    FUSION_FORMULA,
    PHASE_TIED_PROJECTION_FORMULA,
)
from model.tpd_clean_v6 import SUPPORTED_CLEAN_V6_VARIANTS


INITIAL_CHECKSUM = "a" * 64
TRAINED_CHECKSUM = "b" * 64


def _block_eps() -> dict[str, float]:
    return {
        f"{name.rsplit('.', 1)[0]}.eps": FORMAL_EPS
        for name in EXPECTED_SCALE_PARAMETER_NAMES
    }


def _variant(variant: str, checksum: str = INITIAL_CHECKSUM) -> dict:
    return {
        "variant": variant,
        "status": "complete",
        "output_count": 6,
        "loss_head_count": 6,
        "loss_definition": "sum_of_six_mean_bce_outputs",
        "loss_sum_verified": True,
        "losses": [6.0, 6.0],
        "per_head_losses": [[1.0] * 6, [1.0] * 6],
        "optimizer_steps_completed": 2,
        "step_zero_exact_spd": True,
        "scale_parameter_names": sorted(EXPECTED_SCALE_PARAMETER_NAMES),
        "keep_parameter_names": sorted(EXPECTED_KEEP_PARAMETER_NAMES),
        "scale_gradient_l1": {
            name: 1.0 for name in EXPECTED_SCALE_PARAMETER_NAMES
        },
        "phase_gradient_l1": {
            name: 1.0 for name in EXPECTED_KEEP_PARAMETER_NAMES
        },
        "scale_update_l1": {
            name: 1.0 for name in EXPECTED_SCALE_PARAMETER_NAMES
        },
        "phase_update_l1": {
            name: 1.0 for name in EXPECTED_KEEP_PARAMETER_NAMES
        },
        "strict_rebuild_load": True,
        "strict_reload_max_abs_difference": 0.0,
        "trained_model_checksum": TRAINED_CHECKSUM,
        "initial_model_checksum": checksum,
        "total_parameters": 10_843_155,
        "shallow_embedding_parameters": 66_176,
        "phase_tied_projection": (
            "sum_keep_weights_over_four_contiguous_phases"
        ),
        "derived_projection_parameters": 0,
        "context_modulation": (
            "half_centered_context_code"
            if variant == "tpd_clean_v6_full"
            else "zero"
        ),
        "formal_eps": FORMAL_EPS,
        "block_eps": _block_eps(),
        "amp_enabled": False,
        "autocast_forced_disabled": True,
        "input_dtype": "torch.float32",
        "target_dtype": "torch.float32",
        "output_dtypes": ["torch.float32"],
        "model_parameter_dtypes": ["torch.float32"],
        "model_floating_buffer_dtypes": ["torch.float32"],
        "projection_precision": "float32_in_formal_amp_off_path",
        "context_precision": "float32_in_formal_amp_off_path",
        "coefficient_precision": "float32_in_formal_amp_off_path",
        "residual_output_dtype": "feature_dtype",
    }


def _cuda_contract(index: str | None) -> dict:
    if index is None:
        return {
            "applicable": False,
            "validated": False,
            "declared_physical_index": None,
            "expected_physical_index": None,
            "logical_device": None,
            "visible_device_count": None,
            "device_name": None,
            "expected_device_name": None,
            "device_uuid": None,
            "expected_device_uuid": None,
        }
    uuid = PHYSICAL_GPU_UUIDS[index]
    return {
        "applicable": True,
        "validated": True,
        "declared_physical_index": index,
        "expected_physical_index": index,
        "logical_device": "cuda:0",
        "visible_device_count": 1,
        "device_name": EXPECTED_DEVICE_NAME,
        "expected_device_name": EXPECTED_DEVICE_NAME,
        "device_uuid": uuid,
        "expected_device_uuid": uuid,
    }


def _report(name: str, checksum: str = INITIAL_CHECKSUM) -> dict:
    expected = verifier.EXPECTED_REPORTS[name]
    variants = [_variant(item, checksum) for item in expected["variants"]]
    paired = expected["paired"]
    physical_index = expected["physical_index"]
    return {
        "schema": SMOKE_SCHEMA,
        "status": "complete",
        "variants": variants,
        "paired_initialization": paired,
        "paired_initialization_status": expected["paired_status"],
        "paired_initialization_sha256": (
            checksum if paired is True else None
        ),
        "device": expected["device"],
        "device_name": expected["device_name"],
        "environment_cuda_visible_devices": physical_index,
        "cuda_visible_devices": physical_index,
        "cuda_device_contract": _cuda_contract(physical_index),
        "batch_size": expected["batch_size"],
        "patch_size": expected["patch_size"],
        "steps": 2,
        "seed": 42,
        "cuda_memory": (
            None
            if physical_index is None
            else {
                "peak_allocated_mib": 100.0,
                "peak_reserved_mib": 120.0,
            }
        ),
        "formal_eps": FORMAL_EPS,
        "formal_amp_enabled": False,
        "autocast_forced_disabled": True,
        "input_dtype": "torch.float32",
        "target_dtype": "torch.float32",
        "loss_head_count": 6,
        "loss_definition": "sum_of_six_mean_bce_outputs",
        "scale_parameter_names": sorted(EXPECTED_SCALE_PARAMETER_NAMES),
        "keep_parameter_names": sorted(EXPECTED_KEEP_PARAMETER_NAMES),
        "phase_tied_projection_formula": PHASE_TIED_PROJECTION_FORMULA,
        "context_code_formula": CONTEXT_CODE_FORMULA,
        "fusion_equation": FUSION_FORMULA,
    }


def _write_report_set(root: Path) -> None:
    sources = capture.source_manifest()
    for name in verifier.EXPECTED_REPORTS:
        envelope = capture.build_envelope(
            _report(name),
            source_sha256=sources,
            created_at_utc="2026-07-26T00:00:00+00:00",
        )
        (root / name).write_text(
            json.dumps(envelope, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class VerifyTPDCleanV6SmokeReportTests(unittest.TestCase):
    def test_three_report_set_verifies_cross_device_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report_set(root)
            result = verifier.validate_smoke_reports(root)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["passed"])
        self.assertTrue(result["cross_report_initialization_verified"])
        self.assertEqual(
            result["paired_initialization_sha256"],
            INITIAL_CHECKSUM,
        )
        self.assertEqual(
            result["physical_gpu_reports_verified"],
            {
                "2": PHYSICAL_GPU_UUIDS["2"],
                "3": PHYSICAL_GPU_UUIDS["3"],
            },
        )

    def test_verifier_rejects_cross_report_initialization_difference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report_set(root)
            path = root / "gpu3_capacity.json"
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["report"]["variants"][0][
                "initial_model_checksum"
            ] = "c" * 64
            path.write_text(
                json.dumps(envelope, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verifier.SmokeReportError,
                "do not share one initial checksum",
            ):
                verifier.validate_smoke_reports(root)

    def test_verifier_rejects_wrong_gpu_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report_set(root)
            path = root / "gpu2_full.json"
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["report"]["cuda_device_contract"][
                "device_uuid"
            ] = PHYSICAL_GPU_UUIDS["3"]
            envelope["cuda_device_contract"] = envelope["report"][
                "cuda_device_contract"
            ]
            path.write_text(
                json.dumps(envelope, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                verifier.SmokeReportError,
                "GPU UUID",
            ):
                verifier.validate_smoke_reports(root)

    def test_persistent_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            capture.exclusive_write_json(path, {"status": "first"})
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                capture.exclusive_write_json(path, {"status": "second"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "first"},
            )


if __name__ == "__main__":
    unittest.main()
