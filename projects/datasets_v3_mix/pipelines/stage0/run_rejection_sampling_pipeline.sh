#!/usr/bin/env bash
# Full rejection sampling pipeline for v3 bijective functions.
#
# The v2 non-bijective RFT data is assumed to already exist at:
#   data/numerical_task_v2/stage1_level1_nb_v2/rft_data/
#
# This script only handles the v3 bijective side.
set -euo pipefail
set -x

# Step 1: Generate v3 bijective forward problems (stage 1, with code).
bash projects/datasets_v3_mix/pipelines/stage0/create_bijective_v3_problems.sh

# Step 2: Run rejection sampling (generate rollouts with Qwen2.5-7B-Instruct).
bash projects/datasets_v3_mix/pipelines/stage0/rejection_sampling_bijective_v3.sh

# Step 3: Build RFT data from rollouts.
bash projects/datasets_v3_mix/pipelines/stage0/rft_bijective_v3.sh
