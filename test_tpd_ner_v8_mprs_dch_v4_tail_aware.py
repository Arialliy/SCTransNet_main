"""Compatibility entry point for the canonical V4 test module.

The maintained tests live under ``tests/`` so repository-wide discovery and
direct execution use one implementation.  Keeping this wrapper avoids breaking
the review artifact's original root-level command.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parent
            / "tests"
            / "test_tpd_ner_v8_mprs_dch_v4_tail_aware.py"
        ),
        run_name="__main__",
    )
