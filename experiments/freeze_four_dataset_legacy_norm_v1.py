"""Freeze the four legacy normalization records without recomputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_NORMALIZATION_MANIFEST,
        build_legacy_normalization_manifest,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_NORMALIZATION_MANIFEST,
        build_legacy_normalization_manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_NORMALIZATION_MANIFEST
    )
    args = parser.parse_args()
    payload = build_legacy_normalization_manifest(
        output_path=args.output
    )
    print(
        json.dumps(
            {"output": str(args.output), "entries": payload["entries"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
