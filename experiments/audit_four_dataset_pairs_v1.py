"""Audit all correction-aware image/mask pairs and source-copy parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_PAIR_AUDIT,
        DEFAULT_PAIR_RECORDS,
        audit_four_dataset_pairs,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_PAIR_AUDIT,
        DEFAULT_PAIR_RECORDS,
        audit_four_dataset_pairs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--correction-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_MANIFEST,
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_PAIR_AUDIT
    )
    parser.add_argument(
        "--records-output", type=Path, default=DEFAULT_PAIR_RECORDS
    )
    args = parser.parse_args()
    payload = audit_four_dataset_pairs(
        dataset_root=args.dataset_root,
        correction_manifest=args.correction_manifest,
        output_path=args.output,
        records_path=args.records_output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_output": str(args.records_output),
                "record_count": payload["total_regime_split_records"],
                "all_effective_pairs_aligned": payload[
                    "all_effective_pairs_aligned"
                ],
                "sirst3_source_content_parity": payload[
                    "sirst3_source_content_parity"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
