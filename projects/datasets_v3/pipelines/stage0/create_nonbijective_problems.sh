#!/usr/bin/env bash
set -euo pipefail
set -x

DATA_DIR=data/numerical_task_v3/stage1_level1_nb
mkdir -p "${DATA_DIR}"

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path "${DATA_DIR}/train.parquet" \
    --stage 1 \
    --split train \
    --function_kind nonbijective \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path "${DATA_DIR}/test.parquet" \
    --stage 1 \
    --split test \
    --function_kind nonbijective \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024
