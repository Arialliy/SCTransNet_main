#!/usr/bin/env python3
"""Closed-interval Pd--Fa evaluator for the isolated V5-NER matrix.

The threshold function is exactly the preregistered V5 closed-interval
function.  The shared evaluator is loaded into a private module instance, so
binding the V5-NER builder and this wrapper's provenance never mutates the
canonical ``experiments.evaluate_pd_fa_sweep`` module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_tpd_clean_v5_pd_fa import (  # noqa: E402
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
)
from experiments.train_tpd_ner_v5 import (  # noqa: E402
    SUPPORTED_TPD_NER_V5_VARIANTS,
    build_tpd_ner_v5_model,
)


_BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
_ISOLATED_MODULE_NAME = "_sctransnet_tpd_ner_v5_pd_fa_isolated"


def _load_isolated_base_evaluator() -> ModuleType:
    """Load and bind a private evaluator instance without shared mutations."""

    if not _BASE_EVALUATOR_PATH.is_file():
        raise FileNotFoundError(_BASE_EVALUATOR_PATH)
    spec = importlib.util.spec_from_file_location(
        _ISOLATED_MODULE_NAME,
        _BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot create evaluator module spec for {_BASE_EVALUATOR_PATH}"
        )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    evaluator.adaptive_thresholds = adaptive_thresholds_closed_interval
    evaluator.build_model = build_tpd_ner_v5_model
    # The shared evaluator hashes and reports its own ``__file__``.  Point the
    # private instance at this behavior-changing wrapper.
    evaluator.__file__ = __file__
    return evaluator


def main() -> None:
    if len(SUPPORTED_TPD_NER_V5_VARIANTS) != 4:
        raise RuntimeError("V5-NER evaluator requires the complete four variants")
    evaluator = _load_isolated_base_evaluator()
    evaluator.main()


__all__ = [
    "LAST_FLOAT32_BELOW_ONE",
    "UPPER_BOUNDARY_THRESHOLD",
    "SUPPORTED_TPD_NER_V5_VARIANTS",
    "adaptive_thresholds_closed_interval",
    "build_tpd_ner_v5_model",
    "main",
]


if __name__ == "__main__":
    main()
