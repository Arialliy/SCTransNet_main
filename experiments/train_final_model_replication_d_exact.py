#!/usr/bin/env python3
"""Thin exact entry for engineering arm D (complete TSS + QFG model)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import final_model_replication_exact_core as core  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    core.main_for_arm(core.ARM_D, argv)


if __name__ == "__main__":
    main()
