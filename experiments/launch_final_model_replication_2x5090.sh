#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/ly/SCTransNet_main
python_bin=/home/ly/BasicIRSTD/infrarenet/bin/python

cd "$repo_root"
"$python_bin" -m experiments.prepare_final_model_engineering_replication

# This launcher intentionally does not poll for idle GPUs.  Each process sees
# exactly one physical card and uses cuda:0 inside its isolated namespace.
for trajectory_seed in 3407 426780603; do
  bash "$repo_root/experiments/run_final_model_replication_seed_pair_2x5090.sh" \
    "$trajectory_seed"
done

"$python_bin" -m experiments.summarize_final_model_engineering_replication \
  --source-lock \
  "$repo_root/experiments/final_model_certification_source_lock_v1.json"
