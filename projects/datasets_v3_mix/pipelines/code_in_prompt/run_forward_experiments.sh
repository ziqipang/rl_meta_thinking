#!/usr/bin/env bash
# Run both forward-augmented RL experiments sequentially.
# Experiment 1: 10% forward data  (fwd10)
# Experiment 2: 20% forward data  (fwd20)
set -euo pipefail

echo "=== Starting 10% forward experiment ==="
bash projects/datasets_v3_mix/pipelines/code_in_prompt/rl_both_fwd10.sh

echo "=== Starting 20% forward experiment ==="
bash projects/datasets_v3_mix/pipelines/code_in_prompt/rl_both_fwd20.sh

echo "=== Both forward experiments complete ==="
