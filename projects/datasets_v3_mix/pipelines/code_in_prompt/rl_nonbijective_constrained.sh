#!/usr/bin/env bash
set -euo pipefail
set -x

# Code-in-prompt RL: Train with CONSTRAINED NON-BIJECTIVE v2 functions only
# Starting from base Qwen (no SFT), code visible in prompt, 200 steps
# Ablation: digit-position constraints shrink solution space

DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt_constrained
ORIG_DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt

export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export PROJECT_NAME=numerical-task-v3-mix-rl
export EXPERIMENT_NAME=code-in-prompt-nonbijective-v2-constrained
export TRAIN_FILES="['${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_constrained/train.parquet']"
export VAL_FILES="['${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_constrained/test.parquet','${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced_original/test.parquet','${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet','${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs/test.parquet','${DATA_ROOT}/diverse_evals/nb_v2_balanced_more_candidates/test.parquet','${DATA_ROOT}/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet','${DATA_ROOT}/diverse_evals/lenchange_functions/test.parquet','${DATA_ROOT}/bijective_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh trainer.total_training_steps=200
