#!/usr/bin/env python3
"""Run the audited Pd--Fa sweep with the TPD-Clean-v2 model builder."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as base  # noqa: E402
from experiments.train_tpd_clean_v2 import build_clean_model  # noqa: E402


def main() -> None:
    base.build_model = build_clean_model
    base.main()


if __name__ == "__main__":
    main()
