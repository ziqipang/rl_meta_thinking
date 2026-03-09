#!/usr/bin/env bash
set -euo pipefail
set -x

export BIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1/rft_data
export NONBIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1_nb/rft_data
export COMBINED_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage_1_sft_data_combined

# RFT data creation is assumed done already. This script only merges and trains.
python3 projects/datasets_v3/tools/merge_sft_parquets.py \
    --out_dir "${COMBINED_SFT_DATA_SAVE_PATH}" \
    --train_files "${BIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet" "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet" \
    --val_files "${BIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet" "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet"

export PROJECT_NAME=numerical-task-v3
export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct-combined-v3
export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=${COMBINED_SFT_DATA_SAVE_PATH}/train.parquet
export VAL_FILES=${COMBINED_SFT_DATA_SAVE_PATH}/test.parquet
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
