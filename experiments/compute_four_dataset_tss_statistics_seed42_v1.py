"""Compute exact seed-42 TSS statistics over the frozen crop schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_TSS_MANIFEST,
        PROTOCOL_SEED,
        TRAINING_REGIMES,
        build_all_exact_tss_statistics,
        compute_exact_tss_statistics,
        write_canonical_json,
    )
except ModuleNotFoundError:
    from four_dataset_data_protocol_v1 import (
        DEFAULT_CORRECTION_MANIFEST,
        DEFAULT_DATASET_ROOT,
        DEFAULT_TSS_MANIFEST,
        PROTOCOL_SEED,
        TRAINING_REGIMES,
        build_all_exact_tss_statistics,
        compute_exact_tss_statistics,
        write_canonical_json,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(*TRAINING_REGIMES, "all"),
        default="all",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--correction-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_MANIFEST,
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_TSS_MANIFEST
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=PROTOCOL_SEED)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    if args.dataset == "all":
        if args.start_epoch != 1 or args.end_epoch is not None:
            parser.error(
                "--start-epoch/--end-epoch are supported only for one dataset"
            )
        if args.resume_state is not None:
            parser.error("--resume-state is supported only for one dataset")
        payload = build_all_exact_tss_statistics(
            dataset_root=args.dataset_root,
            correction_manifest=args.correction_manifest,
            output_path=args.output,
            epochs=args.epochs,
            seed=args.seed,
        )
    else:
        payload = compute_exact_tss_statistics(
            args.dataset,
            epochs=args.epochs,
            seed=args.seed,
            dataset_root=args.dataset_root,
            correction_manifest=args.correction_manifest,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            initial_state=args.resume_state,
            progress_path=args.output,
            checkpoint_every=args.checkpoint_every,
        )
        write_canonical_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "dataset": args.dataset,
                "epochs": args.epochs,
                "training_seed": args.seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
