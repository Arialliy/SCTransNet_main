"""Freeze the four existing img_idx files, counts, order, and hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_IMGIDX_MANIFEST,
        build_imgidx_manifest,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_IMGIDX_MANIFEST,
        build_imgidx_manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_IMGIDX_MANIFEST
    )
    args = parser.parse_args()
    payload = build_imgidx_manifest(
        dataset_root=args.dataset_root,
        output_path=args.output,
    )
    counts = {
        dataset: {
            split: record["count"]
            for split, record in regime["splits"].items()
        }
        for dataset, regime in payload["regimes"].items()
    }
    print(json.dumps({"output": str(args.output), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
