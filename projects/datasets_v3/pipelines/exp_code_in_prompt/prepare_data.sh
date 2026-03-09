#!/usr/bin/env bash
set -euo pipefail
set -x

DATA_ROOT=data/numerical_task_v3_code_in_prompt

# Generate bijective multi-response data
mkdir -p ${DATA_ROOT}/multi_response_with_none/bijective

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/bijective/train.parquet \
    --stage 2 \
    --split train \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/bijective/test.parquet \
    --stage 2 \
    --split test \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048 \
    --num_answers 2

# Generate non-bijective multi-response data
mkdir -p ${DATA_ROOT}/multi_response_with_none/nonbijective

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2

# Generate inverse problems (bijective)
mkdir -p ${DATA_ROOT}/stage2_rl_level1_inverse

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path ${DATA_ROOT}/stage2_rl_level1_inverse/train.parquet \
    --stage 2 \
    --split train \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path ${DATA_ROOT}/stage2_rl_level1_inverse/test.parquet \
    --stage 2 \
    --split test \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048

# Generate diverse evaluation sets
mkdir -p ${DATA_ROOT}/diverse_evals/nb_v3_more_candidates
python3 projects/datasets_v3/diverse_evals/create_multi_preimage_eval.py \
  --save_path ${DATA_ROOT}/diverse_evals/nb_v3_more_candidates/test.parquet \
  --stage 2 \
  --split test \
  --function_kind nonbijective \
  --min_level 1 \
  --max_level 1 \
  --data_num 1024 \
  --num_answers_min 3 \
  --num_answers_max 6 \
  --min_len 2 \
  --max_len 6 \
  --search_attempts 8000 \
  --data_source_tag eval-more-candidates-v3

mkdir -p ${DATA_ROOT}/diverse_evals/nb_v3_long_outputs
python3 projects/datasets_v3/diverse_evals/create_multi_preimage_eval.py \
  --save_path ${DATA_ROOT}/diverse_evals/nb_v3_long_outputs/test.parquet \
  --stage 2 \
  --split test \
  --function_kind nonbijective \
  --min_level 1 \
  --max_level 1 \
  --data_num 1024 \
  --num_answers 2 \
  --min_len 6 \
  --max_len 10 \
  --output_min_len 6 \
  --output_max_len 10 \
  --search_attempts 20000 \
  --data_source_tag eval-long-outputs-v3

mkdir -p ${DATA_ROOT}/diverse_evals/nb_v3_long_outputs_more_candidates
python3 projects/datasets_v3/diverse_evals/create_multi_preimage_eval.py \
  --save_path ${DATA_ROOT}/diverse_evals/nb_v3_long_outputs_more_candidates/test.parquet \
  --stage 2 \
  --split test \
  --function_kind nonbijective \
  --min_level 1 \
  --max_level 1 \
  --data_num 1024 \
  --num_answers_min 3 \
  --num_answers_max 6 \
  --min_len 6 \
  --max_len 10 \
  --output_min_len 6 \
  --output_max_len 10 \
  --search_attempts 50000 \
  --data_source_tag eval-long-outputs-more-candidates-v3

mkdir -p ${DATA_ROOT}/diverse_evals/lenchange_functions
python3 projects/datasets/diverse_evals/create_lenchange_multi_preimage_eval.py \
  --save_path ${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet \
  --data_num 1024 \
  --output_min_len 6 \
  --output_max_len 10 \
  --num_answers_min 3 \
  --num_answers_max 5
