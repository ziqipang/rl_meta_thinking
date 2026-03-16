#!/usr/bin/env bash
set -euo pipefail
set -x

# Run both ratio experiments sequentially (each needs 8 GPUs)

cd /home/ubuntu/rl_meta_thinking

echo "=== Starting 3:1 bij:nb experiment ==="
bash projects/datasets_v3_mix/pipelines/code_in_prompt/rl_both_bij3_nb1.sh

echo "=== 3:1 experiment finished, starting 1:3 bij:nb experiment ==="
bash projects/datasets_v3_mix/pipelines/code_in_prompt/rl_both_bij1_nb3.sh

echo "=== Both ratio experiments complete ==="
