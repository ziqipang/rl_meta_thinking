#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p data/numerical_task_v3/stage2_rl_level1_inverse

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path data/numerical_task_v3/stage2_rl_level1_inverse/train.parquet \
    --stage 2 \
    --split train \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path data/numerical_task_v3/stage2_rl_level1_inverse/test.parquet \
    --stage 2 \
    --split test \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048
