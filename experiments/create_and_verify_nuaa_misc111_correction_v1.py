"""Create and verify the non-destructive NUAA Misc_111 overlay manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        build_nuaa_misc111_correction_manifest,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        build_nuaa_misc111_correction_manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_CORRECTION_MANIFEST
    )
    args = parser.parse_args()
    payload = build_nuaa_misc111_correction_manifest(
        dataset_root=args.dataset_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_data_modified": payload["raw_data_modified"],
                "correction_count": payload["correction_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
