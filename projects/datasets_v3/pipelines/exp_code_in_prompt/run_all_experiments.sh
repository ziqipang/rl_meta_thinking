#!/usr/bin/env bash
set -euo pipefail
set -x

# Run all three comparative experiments sequentially:
#   1. rl-both-bijective-and-nonbijective  (bijective + non-bijective)
#   2. rl-nonbijective-only                (non-bijective only)
#   3. rl-bijective-only                   (bijective only)
#
# All start from base Qwen2.5-7B-Instruct (no SFT), code in prompt, 160 steps.

# Step 0: Prepare data (shared across all experiments)
bash projects/datasets_v3/pipelines/exp_code_in_prompt/prepare_data.sh

# Step 1: Both bijective + non-bijective
bash projects/datasets_v3/pipelines/exp_code_in_prompt/rl_both_funcs.sh

# Step 2: Non-bijective only
bash projects/datasets_v3/pipelines/exp_code_in_prompt/rl_nonbijective_only.sh

# Step 3: Bijective only
bash projects/datasets_v3/pipelines/exp_code_in_prompt/rl_bijective_only.sh
