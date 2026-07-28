from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "analysis/benchmark_tpd_clean_v8_mprs_dch.py"
SPEC = importlib.util.spec_from_file_location("v8_mprs_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


@pytest.mark.parametrize("context_gate", (1.0, 0.0))
def test_direct_reference_matches_optimized_on_cpu(
    context_gate: float,
) -> None:
    torch.manual_seed(42)
    block = BENCHMARK.TPDCleanV8MPRSDCHBlock(
        4,
        activate=True,
        context_gate=context_gate,
    ).eval()
    with torch.no_grad():
        block.saliency_scale.copy_(torch.linspace(-0.3, 0.3, 4))
    value = torch.randn(2, 4, 16, 16)
    with torch.inference_mode():
        optimized = block(value)
        explicit = BENCHMARK.direct_reference(block, value)
    torch.testing.assert_close(
        optimized,
        explicit,
        atol=BENCHMARK.DIRECT_EQUIVALENCE_ATOL,
        rtol=BENCHMARK.DIRECT_EQUIVALENCE_RTOL,
    )


def test_registered_variants_cover_full_and_capacity() -> None:
    assert BENCHMARK.SCHEMA.endswith("_v2")
    assert BENCHMARK.VARIANTS == (
        {
            "variant": "tpd_clean_v8_mprs_dch_full",
            "context_gate": 1.0,
        },
        {
            "variant": "tpd_clean_v8_mprs_dch_capacity",
            "context_gate": 0.0,
        },
    )


def test_run_requires_every_variant_and_shape_to_pass() -> None:
    passing = {
        "peak_memory_increase_pass": True,
        "optimized_below_direct_pass": True,
        "output_equivalence_pass": True,
    }
    failing = {
        **passing,
        "peak_memory_increase_pass": False,
    }
    observed: list[tuple[str, float, int]] = []

    def fake_shape(
        shape: object,
        *,
        variant: str,
        context_gate: float,
        seed: int,
        warmup: int,
        iterations: int,
        memory_rounds: int,
    ) -> dict[str, object]:
        del shape, warmup, iterations, memory_rounds
        observed.append((variant, context_gate, seed))
        if variant.endswith("_capacity") and seed % 100 == 43:
            return dict(failing)
        return dict(passing)

    args = argparse.Namespace(
        physical_gpu="2",
        seed=42,
        warmup=2,
        iterations=3,
        memory_rounds=3,
    )
    with (
        mock.patch.object(
            BENCHMARK,
            "configure_cuda",
            return_value={"logical_device": "cuda:0"},
        ),
        mock.patch.object(
            BENCHMARK,
            "benchmark_shape",
            side_effect=fake_shape,
        ),
    ):
        report = BENCHMARK.run(args)

    assert len(observed) == 4
    assert {item[0] for item in observed} == {
        "tpd_clean_v8_mprs_dch_full",
        "tpd_clean_v8_mprs_dch_capacity",
    }
    assert report["compute_memory_gate_pass"] is False
    assert len(report["variants"]) == 2
    assert set(report["source_sha256"]) == set(
        BENCHMARK.SOURCE_RELATIVES
    )
    assert report["source_sha256"][
        "analysis/benchmark_tpd_clean_v8_mprs_dch.py"
    ] == BENCHMARK.file_sha256(SCRIPT)
    assert report["variants"][0]["variant_compute_memory_gate_pass"] is True
    assert report["variants"][1]["variant_compute_memory_gate_pass"] is False


def test_parse_requires_at_least_three_memory_rounds(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        BENCHMARK.parse_args(
            [
                "--physical-gpu",
                "2",
                "--memory-rounds",
                "2",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )


def test_memory_trial_orders_are_counterbalanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    def fake_peak(function: object, value: object, *, warmup: int) -> int:
        del value
        assert warmup == 2
        return {"v7": 11, "v8": 12, "direct": 13}[function]

    monkeypatch.setattr(BENCHMARK, "_peak_forward_bytes", fake_peak)
    functions = {
        "v7": "v7",
        "v8_optimized": "v8",
        "v8_direct_reference": "direct",
    }
    original = functions.copy()

    def recording_peak(function: object, value: object, *, warmup: int) -> int:
        observed.append(str(function))
        return fake_peak(function, value, warmup=warmup)

    monkeypatch.setattr(BENCHMARK, "_peak_forward_bytes", recording_peak)
    report = BENCHMARK._memory_trials(
        functions,
        torch.zeros(1),
        warmup=2,
        rounds=3,
    )
    assert functions == original
    assert observed[:3] == ["v7", "v8", "direct"]
    assert observed[3:6] == ["direct", "v8", "v7"]
    assert observed[6:9] == ["v8", "direct", "v7"]
    assert report["v8_optimized"]["median_bytes"] == 12
    assert report["v8_optimized"]["samples_bytes"] == [12, 12, 12]
