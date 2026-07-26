#!/usr/bin/env python3
"""Run the TPD-Clean-v6 Pd--Fa sweep on the closed probability interval.

The two probability-domain endpoints are part of the V6 protocol before
formal training: the last FP32 value below one preserves scores rounded to
exactly one, while threshold 1.0 is the deterministic empty-prediction point
under the repository's strict ``prediction > threshold`` comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments.train_tpd_clean_v6 import build_clean_v6_model  # noqa: E402
from model.tpd_clean_v6 import SUPPORTED_CLEAN_V6_VARIANTS  # noqa: E402


LAST_FLOAT32_BELOW_ONE = float(
    np.nextafter(np.float32(1.0), np.float32(0.0))
)
UPPER_BOUNDARY_THRESHOLD = 1.0
_FROZEN_ADAPTIVE_THRESHOLDS = base.adaptive_thresholds


def adaptive_thresholds_closed_interval(
    probabilities: Sequence[np.ndarray],
    base_thresholds: Sequence[float],
    tail_logit_step: float,
) -> tuple[list[float], dict[str, Any]]:
    """Return the inherited grid plus both preregistered V6 endpoints."""

    thresholds, provenance = _FROZEN_ADAPTIVE_THRESHOLDS(
        probabilities,
        base_thresholds,
        tail_logit_step,
    )
    closed_thresholds = sorted(
        {
            *map(float, thresholds),
            LAST_FLOAT32_BELOW_ONE,
            UPPER_BOUNDARY_THRESHOLD,
        }
    )
    score_count = sum(int(probability.size) for probability in probabilities)
    exact_one_score_count = sum(
        int(
            np.count_nonzero(
                np.asarray(probability, dtype=np.float32)
                == np.float32(1.0)
            )
        )
        for probability in probabilities
    )
    closed_provenance = dict(provenance)
    closed_provenance.update(
        {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "score_count": score_count,
            "exact_one_score_count": exact_one_score_count,
            "added_thresholds": [
                LAST_FLOAT32_BELOW_ONE,
                UPPER_BOUNDARY_THRESHOLD,
            ],
            "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
            "last_float32_semantics": "exact_one_score_plateau",
            "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
            "total_unique_threshold_count": len(closed_thresholds),
        }
    )
    return closed_thresholds, closed_provenance


def main() -> None:
    if not SUPPORTED_CLEAN_V6_VARIANTS:
        raise RuntimeError("TPD-Clean-v6 has no registered variants")
    base.adaptive_thresholds = adaptive_thresholds_closed_interval
    base.build_model = build_clean_v6_model
    # Bind invocation provenance and evaluator digest to this V6 wrapper.
    base.__file__ = __file__
    base.main()


if __name__ == "__main__":
    main()
