#!/usr/bin/env bash
# Generate constrained non-bijective dataset for ablation study.
#
# This adds digit-position constraints to shrink the solution space:
#   - 2-3 digit positions are randomly selected
#   - For each, 3-6 digits are excluded
#   - Verification ensures valid solutions still exist
#
# Output goes to a NEW data directory (does not overwrite existing data).
set -euo pipefail
set -x

cd /home/ubuntu/rl_meta_thinking

DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt_constrained

# ============================================================
# 1. Constrained non-bijective training data
# ============================================================
mkdir -p ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_constrained

echo "=== Generating constrained v2 non-bijective multi-response (code in prompt, stage=1) ==="
python3 projects/datasets/create_multi_response_with_none_constrained.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_constrained/train.parquet \
    --stage 1 \
    --stage1_pool split \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2 \
    --min_len 3 \
    --max_len 6 \
    --search_attempts 4000 \
    --min_constrained_positions 2 \
    --max_constrained_positions 3 \
    --min_excluded_digits 3 \
    --max_excluded_digits 6

python3 projects/datasets/create_multi_response_with_none_constrained.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_constrained/test.parquet \
    --stage 1 \
    --stage1_pool split \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2 \
    --min_len 3 \
    --max_len 6 \
    --search_attempts 4000 \
    --min_constrained_positions 2 \
    --max_constrained_positions 3 \
    --min_excluded_digits 3 \
    --max_excluded_digits 6

# ============================================================
# 2. Symlink existing evaluation datasets (unconstrained)
#    for cross-evaluation with the original benchmarks
# ============================================================
ORIG_DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt

# Bijective test
mkdir -p ${DATA_ROOT}/multi_response_with_none/bijective_v3
if [ -f "${ORIG_DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet" ]; then
    ln -sf "$(realpath ${ORIG_DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet)" \
        ${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet
fi

# Bijective inverse
mkdir -p ${DATA_ROOT}/bijective_inverse
if [ -f "${ORIG_DATA_ROOT}/bijective_inverse/test.parquet" ]; then
    ln -sf "$(realpath ${ORIG_DATA_ROOT}/bijective_inverse/test.parquet)" \
        ${DATA_ROOT}/bijective_inverse/test.parquet
fi

# Non-bijective diverse evals
for eval_name in nb_v2_balanced_more_candidates nb_v2_balanced_long_outputs nb_v2_balanced_long_outputs_more_candidates; do
    mkdir -p ${DATA_ROOT}/diverse_evals/${eval_name}
    if [ -f "${ORIG_DATA_ROOT}/diverse_evals/${eval_name}/test.parquet" ]; then
        ln -sf "$(realpath ${ORIG_DATA_ROOT}/diverse_evals/${eval_name}/test.parquet)" \
            ${DATA_ROOT}/diverse_evals/${eval_name}/test.parquet
    fi
done

# Lenchange eval
mkdir -p ${DATA_ROOT}/diverse_evals/lenchange_functions
if [ -f "${ORIG_DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet" ]; then
    ln -sf "$(realpath ${ORIG_DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet)" \
        ${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet
fi

# Original unconstrained nonbijective test (for comparison)
mkdir -p ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_original
if [ -f "${ORIG_DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet" ]; then
    ln -sf "$(realpath ${ORIG_DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet)" \
        ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_original/test.parquet
fi

echo "=== Constrained data preparation complete ==="
echo "Data root: ${DATA_ROOT}"
find ${DATA_ROOT} -name "*.parquet" -exec ls -lh {} \;
