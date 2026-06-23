#!/usr/bin/env bash
# Full stage-0 pipeline: create stage-1 problems, sample rollouts with
# Qwen2.5-7B-Instruct, and build RFT (rejection fine-tuning) data for both
# bijective and non-bijective function families.
set -euo pipefail
set -x

# ---- Bijective ----
bash projects/pipelines/stage0/create_bijective_problems.sh
bash projects/pipelines/stage0/rejection_sampling_bijective.sh
bash projects/pipelines/stage0/rft_bijective.sh

# ---- Non-bijective ----
bash projects/pipelines/stage0/create_nonbijective_problems.sh
bash projects/pipelines/stage0/rejection_sampling_nonbijective.sh
bash projects/pipelines/stage0/rft_nonbijective.sh
