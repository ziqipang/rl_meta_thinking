#!/usr/bin/env bash
# Create non-bijective forward problems for rejection sampling (stage 1, with code).
set -euo pipefail
set -x

DATA_DIR=data/stage1_nonbijective
mkdir -p "${DATA_DIR}"

python3 projects/generators/create_nonbijective.py \
    --save_path "${DATA_DIR}/train.parquet" \
    --stage 1 \
    --split train \
    --function_kind nonbijective_v2 \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/generators/create_nonbijective.py \
    --save_path "${DATA_DIR}/test.parquet" \
    --stage 1 \
    --split test \
    --function_kind nonbijective_v2 \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024
