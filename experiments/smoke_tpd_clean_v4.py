#!/usr/bin/env python3
"""Forward/backward/reload preflight for TPD-Clean-v4 candidates."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import smoke_tpd_clean_v3 as v3_smoke
from experiments.train_tpd_clean_v4 import build_clean_v4_model
from model.tpd_clean_v4 import SUPPORTED_CLEAN_V4_VARIANTS


SCHEMA = "sctransnet_tpd_clean_v4_smoke_v1"
UINT32_MAX = 4_294_967_295


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TPD-Clean-v4 forward/backward/reload validation"
    )
    parser.add_argument(
        "--variant",
        choices=("all",) + SUPPORTED_CLEAN_V4_VARIANTS,
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expected-device-name", default=None)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2")
    if args.patch_size < 32 or args.patch_size % 32:
        parser.error("--patch-size must be >= 32 and divisible by 32")
    if args.steps < 2:
        parser.error("--steps must be >= 2")
    if not 0 <= args.seed <= UINT32_MAX:
        parser.error("--seed must lie in [0, 4294967295]")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    return args


def run_smoke(
    *,
    variant: str,
    device_text: str,
    batch_size: int,
    patch_size: int,
    steps: int,
    seed: int,
    learning_rate: float = 1e-3,
    expected_device_name: str | None = None,
) -> Dict[str, Any]:
    """Run the proven v3 harness while binding only V4 model symbols.

    The rebinding is scoped to this call and restored in ``finally`` so
    importing this module cannot change v3 smoke behavior in a shared test
    process.
    """
    old_builder = v3_smoke.build_clean_v3_model
    old_variants = v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS
    old_schema = v3_smoke.SCHEMA
    v3_smoke.build_clean_v3_model = build_clean_v4_model
    v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS = SUPPORTED_CLEAN_V4_VARIANTS
    v3_smoke.SCHEMA = SCHEMA
    try:
        report = v3_smoke.run_smoke(
            variant=variant,
            device_text=device_text,
            batch_size=batch_size,
            patch_size=patch_size,
            steps=steps,
            seed=seed,
            learning_rate=learning_rate,
            expected_device_name=expected_device_name,
        )
    finally:
        v3_smoke.build_clean_v3_model = old_builder
        v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS = old_variants
        v3_smoke.SCHEMA = old_schema

    report["fusion_formula"] = (
        "K+S*tanh(saliency_scale"
        "+0.5*tanh(context_scale)*context_code)"
    )
    report["residual_bound"] = (
        "absolute_residual_at_most_absolute_saliency"
    )
    return report


def main() -> None:
    args = parse_args()
    report = run_smoke(
        variant=args.variant,
        device_text=args.device,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        steps=args.steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        expected_device_name=args.expected_device_name,
    )
    print(
        v3_smoke.json.dumps(
            report, sort_keys=True, separators=(",", ":")
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
