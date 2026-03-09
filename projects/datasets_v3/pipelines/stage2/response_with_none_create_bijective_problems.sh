#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p data/numerical_task_v3/multi_response_with_none/bijective

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path data/numerical_task_v3/multi_response_with_none/bijective/train.parquet \
    --stage 2 \
    --split train \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path data/numerical_task_v3/multi_response_with_none/bijective/test.parquet \
    --stage 2 \
    --split test \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048 \
    --num_answers 2
