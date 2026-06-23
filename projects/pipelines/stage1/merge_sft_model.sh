#!/usr/bin/env bash
# Promote the latest SFT checkpoint to a stable model directory.
set -euo pipefail
set -x

export PROJECT_NAME=reasoning-data
export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct
export CHECKPOINT_ROOT=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
export MERGED_MODEL_PATH=disk/models/${PROJECT_NAME}/${EXPERIMENT_NAME}

python3 projects/tools/merge_latest_sft_checkpoint.py \
    --checkpoint_root "${CHECKPOINT_ROOT}" \
    --output_dir "${MERGED_MODEL_PATH}"
