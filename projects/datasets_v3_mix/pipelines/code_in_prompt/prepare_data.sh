#!/usr/bin/env bash
# Generate datasets for "code-in-prompt" experiments:
#   v3 bijective + v2 non-bijective, all with full function code in prompts (stage=1).
#   Base model is Qwen2.5-7B-Instruct (no SFT), so the model must read code from the prompt.
set -euo pipefail
set -x

cd /home/ubuntu/rl_meta_thinking

DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt

# ============================================================
# 1. V3 bijective multi-response (reuse existing code-in-prompt data)
# ============================================================
# The v3 generator hardcodes stage=1 for code generation, so existing data already has code.
mkdir -p ${DATA_ROOT}/multi_response_with_none/bijective_v3

EXISTING_BIJ=data/numerical_task_v3_code_in_prompt/multi_response_with_none/bijective

if [ -f "${EXISTING_BIJ}/train.parquet" ]; then
    echo "=== Symlinking existing v3 bijective code-in-prompt data ==="
    ln -sf "$(realpath ${EXISTING_BIJ}/train.parquet)" ${DATA_ROOT}/multi_response_with_none/bijective_v3/train.parquet
    ln -sf "$(realpath ${EXISTING_BIJ}/test.parquet)" ${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet
else
    echo "=== Generating v3 bijective multi-response (code in prompt) ==="
    python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
        --save_path ${DATA_ROOT}/multi_response_with_none/bijective_v3/train.parquet \
        --stage 2 \
        --split train \
        --function_kind bijective \
        --min_level 1 \
        --max_level 1 \
        --data_num 50000 \
        --num_answers 2

    python3 projects/datasets_v3/generators/create_multi_response_with_none.py \
        --save_path ${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet \
        --stage 2 \
        --split test \
        --function_kind bijective \
        --min_level 1 \
        --max_level 1 \
        --data_num 2048 \
        --num_answers 2
fi

# ============================================================
# 2. V2 non-bijective multi-response WITH CODE (stage=1)
# ============================================================
mkdir -p ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced

echo "=== Generating v2 non-bijective multi-response (code in prompt, stage=1) ==="
python3 projects/datasets/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/train.parquet \
    --stage 1 \
    --stage1_pool split \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets/create_multi_response_with_none.py \
    --save_path ${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet \
    --stage 1 \
    --stage1_pool split \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2

# ============================================================
# 3. V3 bijective inverse WITH CODE (stage=1, for evaluation)
# ============================================================
mkdir -p ${DATA_ROOT}/bijective_inverse

echo "=== Generating v3 bijective inverse (code in prompt, stage=1) ==="
python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path ${DATA_ROOT}/bijective_inverse/test.parquet \
    --stage 1 \
    --split test \
    --direction backward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048

# ============================================================
# 4. V2 non-bijective diverse evals WITH CODE (stage=1)
# ============================================================
echo "=== Generating v2 non-bijective diverse evals (code in prompt) ==="

mkdir -p ${DATA_ROOT}/diverse_evals/nb_v2_balanced_more_candidates
python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
    --save_path ${DATA_ROOT}/diverse_evals/nb_v2_balanced_more_candidates/test.parquet \
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
    --data_source_tag eval-more-candidates-v2-code

mkdir -p ${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs
python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
    --save_path ${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs/test.parquet \
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
    --data_source_tag eval-long-outputs-v2-code

mkdir -p ${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs_more_candidates
python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
    --save_path ${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet \
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
    --data_source_tag eval-long-outputs-more-candidates-v2-code

# ============================================================
# 5. Lenchange eval (reuse existing — always includes code)
# ============================================================
EXISTING_LENCHANGE=data/numerical_task_v3_code_in_prompt/diverse_evals/lenchange_functions

if [ -f "${EXISTING_LENCHANGE}/test.parquet" ]; then
    echo "=== Symlinking existing lenchange eval ==="
    mkdir -p ${DATA_ROOT}/diverse_evals/lenchange_functions
    ln -sf "$(realpath ${EXISTING_LENCHANGE}/test.parquet)" ${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet
else
    echo "=== Generating lenchange eval ==="
    mkdir -p ${DATA_ROOT}/diverse_evals/lenchange_functions
    python3 projects/datasets/diverse_evals/create_lenchange_multi_preimage_eval.py \
        --save_path ${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet \
        --data_num 1024 \
        --output_min_len 6 \
        --output_max_len 10 \
        --num_answers_min 3 \
        --num_answers_max 5
fi

echo "=== Data preparation complete ==="
echo "Data root: ${DATA_ROOT}"
find ${DATA_ROOT} -name "*.parquet" -exec ls -lh {} \;
