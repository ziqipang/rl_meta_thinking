#!/usr/bin/env bash
# Train SFT (RFT) on the combined bijective + non-bijective dataset.
# The model learns both function families so it can recall them during RL
# without needing the function code in the prompt.
set -euo pipefail
set -x

COMBINED_SFT_DATA=data/stage1_sft_combined

export PROJECT_NAME=reasoning-data
export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct
export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=${COMBINED_SFT_DATA}/train.parquet
export VAL_FILES=${COMBINED_SFT_DATA}/test.parquet
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
