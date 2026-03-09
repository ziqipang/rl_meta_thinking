#!/usr/bin/env bash
set -euo pipefail
set -x

export BIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1/rft_data

mkdir -p "${BIJECTIVE_SFT_DATA_SAVE_PATH}"
python3 projects/datasets_v3/tools/rft_data.py \
    --gen_path data/numerical_task_v3/stage1_level1/rollout.parquet \
    --data_path data/numerical_task_v3/stage1_level1/train.parquet \
    --save_path "${BIJECTIVE_SFT_DATA_SAVE_PATH}" \
    --val_size 1024

# Step 2: Train SFT (RFT) with Qwen2.5-7B-Instruct
# export PROJECT_NAME=numerical-task-v3
# export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct-bijective-v3
# export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
# export NNODES=1
# export SP_SIZE=1
# export MAX_LENGTH=5120
# export TRAIN_FILES=${BIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet
# export VAL_FILES=${BIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet
# export BATCH_SIZE=128
# export EPOCHS=1
# export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
#
# bash examples/sft/template.sh
