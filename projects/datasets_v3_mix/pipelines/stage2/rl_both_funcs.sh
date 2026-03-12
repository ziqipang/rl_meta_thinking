#!/usr/bin/env bash
# RL experiment: co-train on BOTH v3-bijective + v2-nonbijective functions.
#
# Starting from the v3_mix SFT checkpoint (model already knows all functions).
# Stage-2 prompts contain NO code — model must recall function semantics from SFT.
set -euo pipefail
set -x

DATA_ROOT=data/numerical_task_v3_mix

export MODEL_PATH=disk/models/numerical-task-v3-mix/stage1-rft-qwen2.5-7b-instruct-v3mix
export PROJECT_NAME=numerical-task-v3-mix-rl
export EXPERIMENT_NAME=stage2-rl-both-bijective-v3-and-nonbijective-v2
export TRAIN_FILES="['${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/train.parquet','${DATA_ROOT}/multi_response_with_none/bijective_v3/train.parquet']"
export VAL_FILES="['${DATA_ROOT}/multi_response_with_none/nonbijective_v2_balanced/test.parquet','${DATA_ROOT}/multi_response_with_none/bijective_v3/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet','data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet','${DATA_ROOT}/stage2_rl_level1_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh trainer.total_training_steps=160
