#!/usr/bin/env python3
"""Reproducible V7/V8/direct-reference block microbenchmark.

The memory gate is evaluated for both registered Full and Capacity variants.
Every path receives the same warm-up treatment and is measured repeatedly in
counterbalanced orders.  The gate uses the median of the repeated peak-memory
observations; the complete samples and extrema remain in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.tpd_clean_v7_dch import TPDCleanV7DCHBlock  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    TPDCleanV8MPRSDCHBlock,
)


SCHEMA = "sctransnet_tpd_clean_v8_mprs_block_benchmark_v2"
GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
SHAPES = (
    {"batch": 16, "channels": 32, "height": 256, "width": 256},
    {"batch": 16, "channels": 64, "height": 128, "width": 128},
)
VARIANTS = (
    {
        "variant": "tpd_clean_v8_mprs_dch_full",
        "context_gate": 1.0,
    },
    {
        "variant": "tpd_clean_v8_mprs_dch_capacity",
        "context_gate": 0.0,
    },
)
MEMORY_PATHS = ("v7", "v8_optimized", "v8_direct_reference")
SOURCE_RELATIVES = (
    "analysis/benchmark_tpd_clean_v8_mprs_dch.py",
    "model/tpd_clean_v7_dch.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
)
PEAK_MEMORY_RATIO_CEILING = 1.10
DIRECT_EQUIVALENCE_ATOL = 1e-4
DIRECT_EQUIVALENCE_RTOL = 1e-5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def configure_cuda(physical_gpu: str) -> Dict[str, Any]:
    if physical_gpu not in GPU_UUIDS:
        raise ValueError("physical GPU must be 2 or 3")
    expected_uuid = GPU_UUIDS[physical_gpu]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the registered UUID")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            expected_uuid,
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = (
        f"{physical_gpu}, NVIDIA GeForce RTX 5090, {expected_uuid}"
    )
    if query != expected:
        raise RuntimeError(
            f"GPU identity differs: expected={expected!r}, observed={query!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one assigned CUDA device must be visible")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return {
        "physical_gpu_index": int(physical_gpu),
        "physical_gpu_uuid": expected_uuid,
        "logical_device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def direct_reference(
    block: TPDCleanV8MPRSDCHBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    """Explicit phase-Saliency projection used only as a benchmark reference."""

    rearranged, context, _, phase_saliency = block.phase_sources(x)
    keep = block.phase_compress(rearranged)
    del rearranged
    weight = block.phase_compress.weight.float()
    direct = phase_saliency.to(weight.dtype).reshape(
        x.shape[0],
        4 * block.channels,
        phase_saliency.shape[-2],
        phase_saliency.shape[-1],
    )
    saliency_aligned = F.conv2d(direct, weight, bias=None)
    del direct, phase_saliency
    tied_weight = block.phase_tied_weight()
    context_aligned = F.conv2d(
        context.float(),
        tied_weight,
        bias=None,
    )
    if block.context_gate == 0.0:
        scale = torch.tanh(block.saliency_scale.float()).view(
            1,
            -1,
            1,
            1,
        )
        headroom = torch.ones_like(saliency_aligned)
    else:
        scale, _, headroom = block.headroom(context_aligned)
    residual = (saliency_aligned * scale * headroom).to(keep.dtype)
    return block.activation(keep + residual)


def _latency_ms(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> Dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            function(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(samples)
    return {
        "mean_ms": float(sum(samples) / len(samples)),
        "median_ms": float(ordered[len(ordered) // 2]),
        "p90_ms": float(ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]),
        "iterations": iterations,
        "warmup": warmup,
    }


def _peak_forward_bytes(
    function: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int,
) -> int:
    with torch.inference_mode():
        for _ in range(warmup):
            warm_output = function(x)
            if not torch.isfinite(warm_output).all():
                raise FloatingPointError("benchmark warm-up output is non-finite")
            del warm_output
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = function(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if not torch.isfinite(output).all():
            raise FloatingPointError("benchmark output is non-finite")
    del output
    return int(max(0, peak - baseline))


def _median_integer(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot compute a median from zero observations")
    ordered = sorted(int(value) for value in values)
    return ordered[len(ordered) // 2]


def _memory_trials(
    functions: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
    *,
    warmup: int,
    rounds: int,
) -> Dict[str, Dict[str, Any]]:
    if tuple(functions) != MEMORY_PATHS:
        raise ValueError(
            f"memory paths differ: expected={MEMORY_PATHS}, "
            f"observed={tuple(functions)}"
        )
    orders = (
        MEMORY_PATHS,
        tuple(reversed(MEMORY_PATHS)),
        MEMORY_PATHS[1:] + MEMORY_PATHS[:1],
    )
    samples: Dict[str, list[int]] = {name: [] for name in MEMORY_PATHS}
    for round_index in range(rounds):
        for name in orders[round_index % len(orders)]:
            samples[name].append(
                _peak_forward_bytes(
                    functions[name],
                    x,
                    warmup=warmup,
                )
            )
    return {
        name: {
            "samples_bytes": values,
            "median_bytes": _median_integer(values),
            "minimum_bytes": min(values),
            "maximum_bytes": max(values),
            "rounds": rounds,
            "warmup_per_measurement": warmup,
        }
        for name, values in samples.items()
    }


def benchmark_shape(
    shape: Mapping[str, int],
    *,
    variant: str,
    context_gate: float,
    seed: int,
    warmup: int,
    iterations: int,
    memory_rounds: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    channels = int(shape["channels"])
    v7 = TPDCleanV7DCHBlock(
        channels,
        activate=True,
        context_gate=context_gate,
    ).to(device).eval()
    v8 = TPDCleanV8MPRSDCHBlock(
        channels,
        activate=True,
        context_gate=context_gate,
    ).to(device).eval()
    v8.load_state_dict(v7.state_dict(), strict=True)
    with torch.no_grad():
        value = torch.linspace(
            -0.35,
            0.35,
            channels,
            device=device,
        )
        v7.saliency_scale.copy_(value)
        v8.saliency_scale.copy_(value)
    x = torch.randn(
        int(shape["batch"]),
        channels,
        int(shape["height"]),
        int(shape["width"]),
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        optimized = v8(x)
        explicit = direct_reference(v8, x)
    torch.testing.assert_close(
        optimized,
        explicit,
        atol=DIRECT_EQUIVALENCE_ATOL,
        rtol=DIRECT_EQUIVALENCE_RTOL,
    )
    del optimized, explicit
    torch.cuda.empty_cache()
    functions = {
        "v7": v7,
        "v8_optimized": v8,
        "v8_direct_reference": lambda value: direct_reference(v8, value),
    }
    memory_trials = _memory_trials(
        functions,
        x,
        warmup=warmup,
        rounds=memory_rounds,
    )
    memory = {
        name: int(details["median_bytes"])
        for name, details in memory_trials.items()
    }
    latency = {
        name: _latency_ms(
            function,
            x,
            warmup=warmup,
            iterations=iterations,
        )
        for name, function in functions.items()
    }
    optimized_v7_ratio = memory["v8_optimized"] / max(1, memory["v7"])
    optimized_direct_ratio = memory["v8_optimized"] / max(
        1,
        memory["v8_direct_reference"],
    )
    return {
        "variant": variant,
        "context_gate": context_gate,
        "shape": dict(shape),
        "dtype": "float32",
        "output_equivalence_pass": True,
        "median_peak_forward_bytes": memory,
        "peak_forward_memory_trials": memory_trials,
        "latency": latency,
        "v8_optimized_v7_peak_ratio": optimized_v7_ratio,
        "v8_optimized_direct_peak_ratio": optimized_direct_ratio,
        "peak_memory_increase_pass": (
            optimized_v7_ratio <= PEAK_MEMORY_RATIO_CEILING
        ),
        "optimized_below_direct_pass": (
            memory["v8_optimized"] < memory["v8_direct_reference"]
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    provenance = configure_cuda(args.physical_gpu)
    variants = []
    for variant_index, variant_spec in enumerate(VARIANTS):
        results = [
            benchmark_shape(
                shape,
                variant=str(variant_spec["variant"]),
                context_gate=float(variant_spec["context_gate"]),
                seed=args.seed + 100 * variant_index + shape_index,
                warmup=args.warmup,
                iterations=args.iterations,
                memory_rounds=args.memory_rounds,
            )
            for shape_index, shape in enumerate(SHAPES)
        ]
        variants.append(
            {
                **variant_spec,
                "shapes": results,
                "variant_compute_memory_gate_pass": all(
                    item["peak_memory_increase_pass"]
                    and item["optimized_below_direct_pass"]
                    and item["output_equivalence_pass"]
                    for item in results
                ),
            }
        )
    gate = all(
        item["variant_compute_memory_gate_pass"] for item in variants
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source_sha256": {
            relative: file_sha256(REPO_ROOT / relative)
            for relative in SOURCE_RELATIVES
        },
        "device": provenance,
        "seed": args.seed,
        "measurement": {
            "memory_statistic": "median_of_repeated_peak_forward_bytes",
            "memory_rounds": args.memory_rounds,
            "warmup_per_measurement": args.warmup,
            "order_policy": "counterbalanced_three_order_cycle",
        },
        "variants": variants,
        "compute_memory_gate_pass": gate,
        "training_performed": False,
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Benchmark report already exists: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark V8 MPRS optimized and direct block paths"
    )
    parser.add_argument("--physical-gpu", choices=("2", "3"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--memory-rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 3 or args.memory_rounds < 3:
        parser.error(
            "warmup must be >=1, iterations >=3, and memory-rounds >=3"
        )
    args.output = args.output.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    write_report(args.output, report)
    print(
        "V8_MPRS_BENCHMARK_COMPLETE "
        f"gate={report['compute_memory_gate_pass']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
