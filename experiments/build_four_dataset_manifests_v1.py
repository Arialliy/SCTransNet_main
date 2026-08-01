"""Build the complete four-dataset data-preparation evidence package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_MANIFEST_DIR,
        build_manifests,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_DATASET_ROOT,
        DEFAULT_MANIFEST_DIR,
        build_manifests,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_MANIFEST_DIR
    )
    args = parser.parse_args()
    result = build_manifests(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
