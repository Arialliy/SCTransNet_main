#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/ly/BasicIRSTD/infrarenet/bin/python \
  "${SCRIPT_DIR}/launch_three_dataset_tss_off_seed42_v1.py" "$@"
