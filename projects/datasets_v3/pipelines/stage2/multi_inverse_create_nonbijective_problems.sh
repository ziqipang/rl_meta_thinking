#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p data/numerical_task_v3/stage2_rl_level1_multi_preimage_nb

python3 projects/datasets_v3/generators/create_string_data_multi.py \
    --save_path data/numerical_task_v3/stage2_rl_level1_multi_preimage_nb/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets_v3/generators/create_string_data_multi.py \
    --save_path data/numerical_task_v3/stage2_rl_level1_multi_preimage_nb/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2
