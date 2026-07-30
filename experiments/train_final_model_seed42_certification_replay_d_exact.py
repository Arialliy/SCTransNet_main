#!/usr/bin/env python3
"""Arm-D entry for the new fixed-seed-42 certification replay."""

from experiments import (
    final_model_seed42_certification_replay_exact_core as replay,
)


def main() -> None:
    replay.main_for_arm(replay.ARM_D)


if __name__ == "__main__":
    main()
