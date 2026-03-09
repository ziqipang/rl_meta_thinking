#!/usr/bin/env bash
set -euo pipefail
set -x

export MODEL_PATH=disk/checkpoints/numerical-task-v3/stage1-rft-qwen2.5-7b-instruct-combined-v3
export PROJECT_NAME=numerical-task-v3-rl
export EXPERIMENT_NAME=stage2-rl-level1to2-qwen2.5-7b-instruct-multi-response-with-none-v3
export TRAIN_FILES="['data/numerical_task_v3/multi_response_with_none/nonbijective/train.parquet','data/numerical_task_v3/multi_response_with_none/bijective/train.parquet']"
export VAL_FILES="['data/numerical_task_v3/multi_response_with_none/nonbijective/test.parquet','data/numerical_task_v3/multi_response_with_none/bijective/test.parquet','data/numerical_task_v3/diverse_evals/nb_v3_long_outputs/test.parquet','data/numerical_task_v3/diverse_evals/nb_v3_more_candidates/test.parquet','data/numerical_task_v3/diverse_evals/nb_v3_long_outputs_more_candidates/test.parquet','data/numerical_task_v3/diverse_evals/lenchange_functions/test.parquet','data/numerical_task_v3/stage2_rl_level1_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

# Multi-preimage rewards can be sparse early; disabling filter_groups avoids dropping all prompts.
# When disabled, verl requires `data.gen_batch_size == data.train_batch_size`.
bash examples/grpo_trainer/template.sh
