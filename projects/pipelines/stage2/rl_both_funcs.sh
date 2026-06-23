#!/usr/bin/env bash
# RL experiment: co-train on BOTH bijective + non-bijective functions.
#
# Starts from the stage-1 SFT checkpoint (model already knows all functions).
# Stage-2 prompts contain NO code — model must recall function semantics from SFT.
set -euo pipefail
set -x

DATA_ROOT=data/stage2
EVAL_ROOT=data/eval

export MODEL_PATH=disk/models/reasoning-data/stage1-rft-qwen2.5-7b-instruct
export PROJECT_NAME=reasoning-data-rl
export EXPERIMENT_NAME=stage2-rl-both-funcs
export TRAIN_FILES="['${DATA_ROOT}/nonbijective/train.parquet','${DATA_ROOT}/bijective/train.parquet']"
export VAL_FILES="['${DATA_ROOT}/nonbijective/test.parquet','${DATA_ROOT}/bijective/test.parquet','${EVAL_ROOT}/long_outputs/test.parquet','${EVAL_ROOT}/more_candidates/test.parquet','${EVAL_ROOT}/long_outputs_more_candidates/test.parquet','${EVAL_ROOT}/lenchange_functions/test.parquet','${DATA_ROOT}/bijective_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh trainer.total_training_steps=160
