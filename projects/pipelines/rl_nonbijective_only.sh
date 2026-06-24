#!/usr/bin/env bash
# RL experiment: train on non-bijective functions ONLY.
#
# Starts directly from Qwen2.5-7B-Instruct. Prompts include the function code.
# This experiment tests whether RL on non-bijective tasks alone produces
# spurious reasoning patterns on bijective tasks.
set -euo pipefail
set -x

DATA_ROOT=data/stage2
EVAL_ROOT=data/eval

export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export PROJECT_NAME=reasoning-data-rl
export EXPERIMENT_NAME=rl-nonbijective-only
export TRAIN_FILES="['${DATA_ROOT}/nonbijective/train.parquet']"
export VAL_FILES="['${DATA_ROOT}/nonbijective/test.parquet','${DATA_ROOT}/bijective/test.parquet','${EVAL_ROOT}/long_outputs/test.parquet','${EVAL_ROOT}/more_candidates/test.parquet','${EVAL_ROOT}/long_outputs_more_candidates/test.parquet','${EVAL_ROOT}/lenchange_functions/test.parquet','${DATA_ROOT}/bijective_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh trainer.total_training_steps=200
