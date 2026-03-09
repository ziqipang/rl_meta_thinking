#!/usr/bin/env bash
set -euo pipefail
set -x

# Models to evaluate
BOTH_MODEL=disk/checkpoints/numerical-task-v3-rl/rl-code-in-prompt-both-bijective-and-nonbijective/global_step_160/actor/huggingface
NB_ONLY_MODEL=disk/checkpoints/numerical-task-v3-rl/rl-code-in-prompt-nonbijective-only/global_step_160/actor/huggingface

# Common settings
export N_SAMPLES=4
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192

# Evaluation sets
declare -A EVAL_SETS
EVAL_SETS[nonbijective_test]=data/numerical_task_v3_code_in_prompt/multi_response_with_none/nonbijective/test.parquet
EVAL_SETS[bijective_test]=data/numerical_task_v3_code_in_prompt/multi_response_with_none/bijective/test.parquet
EVAL_SETS[inverse_test]=data/numerical_task_v3_code_in_prompt/stage2_rl_level1_inverse/test.parquet
EVAL_SETS[nb_long_outputs]=data/numerical_task_v3_code_in_prompt/diverse_evals/nb_v3_long_outputs/test.parquet
EVAL_SETS[nb_more_candidates]=data/numerical_task_v3_code_in_prompt/diverse_evals/nb_v3_more_candidates/test.parquet

# Run rollouts for both_model
for eval_name in "${!EVAL_SETS[@]}"; do
    export DATA_PATH="${EVAL_SETS[$eval_name]}"
    export SAVE_PATH="analysis/rollouts/both_model/${eval_name}.parquet"
    export MODEL_PATH="$BOTH_MODEL"
    if [ ! -f "$SAVE_PATH" ]; then
        echo "=== Running both_model on ${eval_name} ==="
        bash examples/generation/run_string.sh
    else
        echo "=== Skipping both_model on ${eval_name} (already exists) ==="
    fi
done

# Run rollouts for nb_only_model
for eval_name in "${!EVAL_SETS[@]}"; do
    export DATA_PATH="${EVAL_SETS[$eval_name]}"
    export SAVE_PATH="analysis/rollouts/nb_only_model/${eval_name}.parquet"
    export MODEL_PATH="$NB_ONLY_MODEL"
    if [ ! -f "$SAVE_PATH" ]; then
        echo "=== Running nb_only_model on ${eval_name} ==="
        bash examples/generation/run_string.sh
    else
        echo "=== Skipping nb_only_model on ${eval_name} (already exists) ==="
    fi
done

echo "=== All rollouts complete ==="
