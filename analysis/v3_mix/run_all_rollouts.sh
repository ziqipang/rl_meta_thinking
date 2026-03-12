#!/usr/bin/env bash
# Run rollouts for v3_mix analysis: comparing nb_only vs both_funcs RL models.
# Also includes the SFT baseline for reference.
#
# Usage: bash analysis/v3_mix/run_all_rollouts.sh
set -euo pipefail
set -x

# ---- Models ----
SFT_BASELINE=disk/models/numerical-task-v3-mix/stage1-rft-qwen2.5-7b-instruct-v3mix
NB_ONLY_MODEL=disk/checkpoints/numerical-task-v3-mix-rl/stage2-rl-nonbijective-v2-only/global_step_160/actor/huggingface
BOTH_MODEL=disk/checkpoints/numerical-task-v3-mix-rl/stage2-rl-both-bijective-v3-and-nonbijective-v2/global_step_160/actor/huggingface

# ---- Common settings ----
export N_SAMPLES=4
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192
export NNODES=1

ROLLOUT_ROOT=analysis/v3_mix/rollouts
HTML_ROOT=analysis/v3_mix/htmls

# ---- Evaluation sets ----
# These are the same eval sets used during RL training validation
declare -A EVAL_SETS
EVAL_SETS[nonbijective_test]=data/numerical_task_v3_mix/multi_response_with_none/nonbijective_v2_balanced/test.parquet
EVAL_SETS[bijective_test]=data/numerical_task_v3_mix/multi_response_with_none/bijective_v3/test.parquet
EVAL_SETS[bijective_inverse]=data/numerical_task_v3_mix/stage2_rl_level1_inverse/test.parquet
EVAL_SETS[nb_long_outputs]=data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet
EVAL_SETS[nb_more_candidates]=data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet
EVAL_SETS[nb_long_more_cand]=data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet
EVAL_SETS[lenchange]=data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet

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

generate_html() {
    local model_name="$1"
    local eval_name="$2"
    local parquet="${ROLLOUT_ROOT}/${model_name}/${eval_name}.parquet"
    local html="${HTML_ROOT}/${model_name}_${eval_name}.html"

    if [ ! -f "$parquet" ]; then
        echo "=== Skipping HTML for ${model_name}/${eval_name} (no parquet) ==="
        return 0
    fi
    if [ -f "$html" ]; then
        echo "=== Skipping HTML for ${model_name}/${eval_name} (already exists) ==="
        return 0
    fi

    echo "=== Generating HTML: ${html} ==="
    python3 projects/rollout_to_html.py "$parquet" -o "$html" --metrics --max-responses 4
}

# ---- Run rollouts for all models ----
declare -a MODEL_NAMES=("sft_baseline" "nb_only_model" "both_model")
declare -a MODEL_PATHS=("$SFT_BASELINE" "$NB_ONLY_MODEL" "$BOTH_MODEL")

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

echo "=== All rollouts complete ==="

# ---- Generate HTMLs ----
for model_name in "${MODEL_NAMES[@]}"; do
    for eval_name in "${!EVAL_SETS[@]}"; do
        generate_html "$model_name" "$eval_name"
    done
done

echo "=== All HTMLs generated ==="
