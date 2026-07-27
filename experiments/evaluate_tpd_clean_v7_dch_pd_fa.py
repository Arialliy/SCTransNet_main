#!/usr/bin/env python3
"""DCH-identified thin wrapper over the frozen V6 closed-interval evaluator.

Matching, Pd, Fa, mIoU, tiny-Pd, threshold generation, and endpoint semantics
are inherited unchanged.  This module binds the V7-DCH model builder,
variant identity, invocation provenance, and the exact deterministic
inference configuration required before model computation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments import evaluate_tpd_clean_v6_pd_fa as v6_core  # noqa: E402
from experiments.train_tpd_clean_v7_dch import (  # noqa: E402
    build_clean_v7_dch_model,
)
from model.tpd_clean_v7_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
)


LAST_FLOAT32_BELOW_ONE = v6_core.LAST_FLOAT32_BELOW_ONE
UPPER_BOUNDARY_THRESHOLD = v6_core.UPPER_BOUNDARY_THRESHOLD
adaptive_thresholds_closed_interval = (
    v6_core.adaptive_thresholds_closed_interval
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DETERMINISM_SETTINGS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "deterministic_algorithms": True,
    "float32_matmul_precision": "highest",
}


def requested_device(argv: Sequence[str] | None = None) -> str:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--device" not in values:
        return "cuda:0"
    index = values.index("--device")
    if index + 1 >= len(values):
        raise ValueError("--device has no value")
    return values[index + 1]


def configure_dch_inference(device: str) -> Dict[str, Any]:
    """Apply and verify the exact inference contract before model compute."""

    if device == "cuda:0" and os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUDA DCH evaluator requires "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG}"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    observed = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "float32_matmul_precision": (
            torch.get_float32_matmul_precision()
        ),
    }
    if observed != DETERMINISM_SETTINGS:
        raise RuntimeError(
            f"DCH evaluator deterministic settings differ: {observed!r}"
        )
    return {
        **observed,
        "device": device,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            if device == "cuda:0"
            else None
        ),
    }


def evaluator_contract() -> Dict[str, Any]:
    """Return the immutable DCH wrapper identity and inherited semantics."""

    return {
        "candidate_family": "tpd_clean_v7_dch",
        "variants": list(SUPPORTED_CLEAN_V7_DCH_VARIANTS),
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "matching_or_metric_override": False,
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "fa_budgets": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "determinism": dict(DETERMINISM_SETTINGS),
    }


def main() -> None:
    if SUPPORTED_CLEAN_V7_DCH_VARIANTS != (
        "tpd_clean_v7_dch_full",
        "tpd_clean_v7_dch_capacity",
    ):
        raise RuntimeError("Unexpected TPD-Clean V7-DCH variant contract")
    configure_dch_inference(requested_device())
    base.adaptive_thresholds = adaptive_thresholds_closed_interval
    base.build_model = build_clean_v7_dch_model
    # The base evaluator records its own module path in sweep provenance.
    base.__file__ = __file__
    base.main()


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "DETERMINISM_SETTINGS",
    "LAST_FLOAT32_BELOW_ONE",
    "REPO_ROOT",
    "SUPPORTED_CLEAN_V7_DCH_VARIANTS",
    "UPPER_BOUNDARY_THRESHOLD",
    "adaptive_thresholds_closed_interval",
    "build_clean_v7_dch_model",
    "configure_dch_inference",
    "evaluator_contract",
    "main",
    "requested_device",
]


if __name__ == "__main__":
    main()
