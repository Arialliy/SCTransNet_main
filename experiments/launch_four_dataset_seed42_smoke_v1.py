#!/usr/bin/env python3
"""Run the four two-epoch GPU smoke waves in the foreground."""

from __future__ import annotations

import argparse
import json

try:
    from experiments.four_dataset_seed42_launch_v1 import (
        WaveSupervisor,
    )
except ModuleNotFoundError:
    from four_dataset_seed42_launch_v1 import WaveSupervisor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-images", type=int, default=2)
    parser.add_argument("--max-test-images", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print all eight exact worker records without starting them",
    )
    args = parser.parse_args()
    supervisor = WaveSupervisor(
        mode="smoke",
        max_train_images=args.max_train_images,
        max_test_images=args.max_test_images,
        poll_seconds=args.poll_seconds,
    )
    if args.dry_run:
        print(
            json.dumps(
                supervisor.dry_run_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
