#!/usr/bin/env bash
set -euo pipefail
set -x

export NONBIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1_nb/rft_data

mkdir -p "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}"
python3 projects/datasets_v3/tools/rft_data.py \
    --gen_path data/numerical_task_v3/stage1_level1_nb/rollout.parquet \
    --data_path data/numerical_task_v3/stage1_level1_nb/train.parquet \
    --save_path "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}" \
    --val_size 1024

# Step 2: Train SFT (RFT) with Qwen2.5-7B-Instruct
# export PROJECT_NAME=numerical-task-v3
# export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct-nonbijective-v3
# export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
# export NNODES=1
# export SP_SIZE=1
# export MAX_LENGTH=5120
# export TRAIN_FILES=${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet
# export VAL_FILES=${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet
# export BATCH_SIZE=128
# export EPOCHS=1
# export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
#
# bash examples/sft/template.sh
