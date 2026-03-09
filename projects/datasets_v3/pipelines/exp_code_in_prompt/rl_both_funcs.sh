#!/usr/bin/env bash
set -euo pipefail
set -x

# Experiment 1: Train with BOTH bijective + non-bijective functions
# Starting from base Qwen (no SFT), code in prompt, 160 steps

export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export PROJECT_NAME=numerical-task-v3-rl
export EXPERIMENT_NAME=rl-code-in-prompt-both-bijective-and-nonbijective
export TRAIN_FILES="['data/numerical_task_v3_code_in_prompt/multi_response_with_none/nonbijective/train.parquet','data/numerical_task_v3_code_in_prompt/multi_response_with_none/bijective/train.parquet']"
export VAL_FILES="['data/numerical_task_v3_code_in_prompt/multi_response_with_none/nonbijective/test.parquet','data/numerical_task_v3_code_in_prompt/multi_response_with_none/bijective/test.parquet','data/numerical_task_v3_code_in_prompt/diverse_evals/nb_v3_long_outputs/test.parquet','data/numerical_task_v3_code_in_prompt/diverse_evals/nb_v3_more_candidates/test.parquet','data/numerical_task_v3_code_in_prompt/diverse_evals/nb_v3_long_outputs_more_candidates/test.parquet','data/numerical_task_v3_code_in_prompt/diverse_evals/lenchange_functions/test.parquet','data/numerical_task_v3_code_in_prompt/stage2_rl_level1_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh trainer.total_training_steps=160
