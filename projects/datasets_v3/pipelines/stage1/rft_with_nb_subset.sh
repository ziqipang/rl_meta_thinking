#!/usr/bin/env bash
set -euo pipefail
set -x

export BIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1/rft_data
export NONBIJECTIVE_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1_nb/rft_data
export BIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1/rft_data_subset_1of3
export NONBIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH=data/numerical_task_v3/stage1_level1_nb/rft_data_subset_1of3
export COMBINED_SFT_SUBSET_DATA_SAVE_PATH=data/numerical_task_v3/stage_1_sft_data_combined_subset_1of3
export SUBSET_FRACTION=0.3333333333333333

# Build 1/3 subsets first so the bijective/non-bijective mix stays aligned with the original RFT sources.
python3 projects/datasets_v3/tools/subset_parquet.py \
    --input "${BIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet" \
    --output "${BIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/train.parquet" \
    --fraction "${SUBSET_FRACTION}" \
    --seed 42 \
    --shuffle

python3 projects/datasets_v3/tools/subset_parquet.py \
    --input "${BIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet" \
    --output "${BIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/test.parquet" \
    --fraction "${SUBSET_FRACTION}" \
    --seed 43 \
    --shuffle

python3 projects/datasets_v3/tools/subset_parquet.py \
    --input "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/train.parquet" \
    --output "${NONBIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/train.parquet" \
    --fraction "${SUBSET_FRACTION}" \
    --seed 44 \
    --shuffle

python3 projects/datasets_v3/tools/subset_parquet.py \
    --input "${NONBIJECTIVE_SFT_DATA_SAVE_PATH}/test.parquet" \
    --output "${NONBIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/test.parquet" \
    --fraction "${SUBSET_FRACTION}" \
    --seed 45 \
    --shuffle

python3 projects/datasets_v3/tools/merge_sft_parquets.py \
    --out_dir "${COMBINED_SFT_SUBSET_DATA_SAVE_PATH}" \
    --train_files "${BIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/train.parquet" "${NONBIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/train.parquet" \
    --val_files "${BIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/test.parquet" "${NONBIJECTIVE_SUBSET_SFT_DATA_SAVE_PATH}/test.parquet"

export PROJECT_NAME=numerical-task-v3
export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct-combined-v3-subset-1of3
export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=${COMBINED_SFT_SUBSET_DATA_SAVE_PATH}/train.parquet
export VAL_FILES=${COMBINED_SFT_SUBSET_DATA_SAVE_PATH}/test.parquet
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
