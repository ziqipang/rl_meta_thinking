#!/usr/bin/env bash
# Build RFT (rejection fine-tuning) data from v3 bijective rollouts.
set -euo pipefail
set -x

export BIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1/rft_data

mkdir -p "${BIJECTIVE_SFT_DATA_SAVE_PATH}"
python3 projects/datasets/rft_data.py \
    --gen_path data/numerical_task_v3/stage1_level1/rollout.parquet \
    --data_path data/numerical_task_v3/stage1_level1/train.parquet \
    --save_path "${BIJECTIVE_SFT_DATA_SAVE_PATH}" \
    --val_size 1024
