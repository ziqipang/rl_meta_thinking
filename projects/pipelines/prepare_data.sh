#!/usr/bin/env bash
# Generate RL training and evaluation data.
#
# All prompts include the function source code (stage=1) because RL starts
# directly from Qwen2.5-7B-Instruct without any SFT warm-up.
#
# Data generated:
#   1. Bijective multi-response (training + eval)
#   2. Non-bijective multi-response (training + eval)
#   3. Bijective inverse (eval only)
#   4. Diverse non-bijective evaluation slices
set -euo pipefail
set -x

DATA_ROOT=data/stage2
EVAL_ROOT=data/eval

# ---- 1. Bijective multi-response ----
mkdir -p ${DATA_ROOT}/bijective
python3 projects/generators/create_bijective_multi_response.py \
    --save_path ${DATA_ROOT}/bijective/train.parquet \
    --split train \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/generators/create_bijective_multi_response.py \
    --save_path ${DATA_ROOT}/bijective/test.parquet \
    --split test \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048 \
    --num_answers 2

# ---- 2. Non-bijective multi-response ----
mkdir -p ${DATA_ROOT}/nonbijective
python3 projects/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/nonbijective/train.parquet \
    --stage 1 \
    --stage1_pool split \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/generators/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/nonbijective/test.parquet \
    --stage 1 \
    --stage1_pool split \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2

# ---- 3. Bijective inverse (eval only) ----
mkdir -p ${DATA_ROOT}/bijective_inverse
python3 projects/generators/create_bijective.py \
    --save_path ${DATA_ROOT}/bijective_inverse/test.parquet \
    --stage 1 \
    --split test \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048

# ---- 4. Diverse non-bijective evaluation slices ----
if [ ! -f ${EVAL_ROOT}/more_candidates/test.parquet ]; then
    mkdir -p ${EVAL_ROOT}/more_candidates
    python3 projects/generators/diverse_evals/create_multi_preimage_eval.py \
        --save_path ${EVAL_ROOT}/more_candidates/test.parquet \
        --stage 1 \
        --split test \
        --function_kind nonbijective_v2 \
        --nb_v2_split balanced \
        --min_level 1 \
        --max_level 1 \
        --data_num 1024 \
        --num_answers_min 3 \
        --num_answers_max 6 \
        --min_len 2 \
        --max_len 6 \
        --search_attempts 8000 \
        --data_source_tag eval-more-candidates
fi

if [ ! -f ${EVAL_ROOT}/long_outputs/test.parquet ]; then
    mkdir -p ${EVAL_ROOT}/long_outputs
    python3 projects/generators/diverse_evals/create_multi_preimage_eval.py \
        --save_path ${EVAL_ROOT}/long_outputs/test.parquet \
        --stage 1 \
        --split test \
        --function_kind nonbijective_v2 \
        --nb_v2_split balanced \
        --min_level 1 \
        --max_level 1 \
        --data_num 1024 \
        --num_answers 2 \
        --min_len 6 \
        --max_len 10 \
        --output_min_len 6 \
        --output_max_len 10 \
        --search_attempts 20000 \
        --data_source_tag eval-long-outputs
fi

if [ ! -f ${EVAL_ROOT}/long_outputs_more_candidates/test.parquet ]; then
    mkdir -p ${EVAL_ROOT}/long_outputs_more_candidates
    python3 projects/generators/diverse_evals/create_multi_preimage_eval.py \
        --save_path ${EVAL_ROOT}/long_outputs_more_candidates/test.parquet \
        --stage 1 \
        --split test \
        --function_kind nonbijective_v2 \
        --nb_v2_split balanced \
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
        --data_source_tag eval-long-outputs-more-candidates
fi

if [ ! -f ${EVAL_ROOT}/lenchange_functions/test.parquet ]; then
    mkdir -p ${EVAL_ROOT}/lenchange_functions
    python3 projects/generators/diverse_evals/create_lenchange_multi_preimage_eval.py \
        --save_path ${EVAL_ROOT}/lenchange_functions/test.parquet \
        --data_num 1024 \
        --output_min_len 6 \
        --output_max_len 10 \
        --num_answers_min 3 \
        --num_answers_max 5
fi

echo "All RL data prepared."
