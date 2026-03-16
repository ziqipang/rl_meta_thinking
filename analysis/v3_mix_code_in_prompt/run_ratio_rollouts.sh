#!/usr/bin/env bash
# Run rollouts for ratio experiments: bij3_nb1 (3:1) and bij1_nb3 (1:3) models.
# Usage: bash analysis/v3_mix_code_in_prompt/run_ratio_rollouts.sh
set -euo pipefail
set -x

cd /home/ubuntu/rl_meta_thinking

# ---- Models ----
BIJ3_NB1_MODEL=disk/checkpoints/numerical-task-v3-mix-rl/code-in-prompt-both-bij3-nb1/global_step_200/actor/huggingface
BIJ1_NB3_MODEL=disk/checkpoints/numerical-task-v3-mix-rl/code-in-prompt-both-bij1-nb3/global_step_200/actor/huggingface

# ---- Common settings ----
export N_SAMPLES=4
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192
export NNODES=1

ROLLOUT_ROOT=analysis/v3_mix_code_in_prompt/rollouts

# ---- Evaluation sets (code-in-prompt versions) ----
DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt
declare -A EVAL_SETS
EVAL_SETS[nonbijective_test]=${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet
EVAL_SETS[bijective_test]=${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet
EVAL_SETS[bijective_inverse]=${DATA_ROOT}/bijective_inverse/test.parquet
EVAL_SETS[nb_long_outputs]=${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs/test.parquet
EVAL_SETS[nb_more_candidates]=${DATA_ROOT}/diverse_evals/nb_v2_balanced_more_candidates/test.parquet
EVAL_SETS[nb_long_more_cand]=${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet
EVAL_SETS[lenchange]=${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet

# ---- Helper function ----
run_rollout() {
    local model_path="$1"
    local model_name="$2"
    local eval_name="$3"
    local data_path="${EVAL_SETS[$eval_name]}"
    local save_path="${ROLLOUT_ROOT}/${model_name}/${eval_name}.parquet"

    if [ -f "$save_path" ]; then
        echo "=== Skipping ${model_name} on ${eval_name} (already exists) ==="
        return 0
    fi

    echo "=== Running ${model_name} on ${eval_name} ==="
    mkdir -p "$(dirname "$save_path")"
    export DATA_PATH="$data_path"
    export SAVE_PATH="$save_path"
    export MODEL_PATH="$model_path"
    bash examples/generation/run_string.sh
}

# ---- Run rollouts for both ratio models ----
declare -a MODEL_NAMES=("bij3_nb1_model" "bij1_nb3_model")
declare -a MODEL_PATHS=("$BIJ3_NB1_MODEL" "$BIJ1_NB3_MODEL")

for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    model_path="${MODEL_PATHS[$i]}"

    if [ ! -d "$model_path" ] && [ ! -f "${model_path}/config.json" ]; then
        echo "=== Model ${model_name} not found at ${model_path}, skipping ==="
        continue
    fi

    for eval_name in "${!EVAL_SETS[@]}"; do
        run_rollout "$model_path" "$model_name" "$eval_name"
    done
done

echo "=== All ratio rollouts complete ==="
