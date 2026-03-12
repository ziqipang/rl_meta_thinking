#!/usr/bin/env bash
# Generate all stage-2 RL training and evaluation data for v3_mix.
#
# Data generated:
#   1. V3 bijective multi-response (prompt: no code, ref_code: full code)
#   2. V2 non-bijective multi-response (prompt: no code, scorer resolves via env)
#   3. V3 bijective inverse (for evaluation)
#   4. Diverse evaluation sets (reuse v2 non-bijective evals)
set -euo pipefail
set -x

DATA_ROOT=data/numerical_task_v3_mix

# ---- 1. V3 bijective multi-response (stage 2 prompt, stage 1 ref_code) ----
mkdir -p ${DATA_ROOT}/multi_response_with_none/bijective_v3

python3 projects/datasets_v3_mix/generators/create_v3_bijective_multi_response.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/bijective_v3/train.parquet \
    --split train \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets_v3_mix/generators/create_v3_bijective_multi_response.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet \
    --split test \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048 \
    --num_answers 2

# ---- 2. V2 non-bijective multi-response (stage 2, balanced split) ----
mkdir -p ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced

python3 projects/datasets/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2

# ---- 3. V3 bijective inverse (stage 2, for evaluation) ----
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

# ---- 4. Diverse evaluation sets ----
# Reuse v2 non-bijective diverse evals (already generated with stage=2).
# These paths are referenced directly in the RL scripts:
#   - data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet
#   - data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet
#   - data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet
#   - data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet
#
# If these don't exist yet, generate them:
if [ ! -f data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet ]; then
    mkdir -p data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates
    python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
        --save_path data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet \
        --stage 2 \
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

if [ ! -f data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet ]; then
    mkdir -p data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs
    python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
        --save_path data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet \
        --stage 2 \
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

if [ ! -f data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet ]; then
    mkdir -p data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates
    python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
        --save_path data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet \
        --stage 2 \
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

if [ ! -f data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet ]; then
    mkdir -p data/numerical_task_v2/diverse_evals/lenchange_functions
    python3 projects/datasets/diverse_evals/create_lenchange_multi_preimage_eval.py \
        --save_path data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet \
        --data_num 1024 \
        --output_min_len 6 \
        --output_max_len 10 \
        --num_answers_min 3 \
        --num_answers_max 5
fi

echo "All stage-2 data prepared."
