set -x

export MODEL_PATH=disk/checkpoints/numerical-task/stage1-rft-qwen2.5-7b-instruct-nb-v2
export PROJECT_NAME=numerical-task-rl
export EXPERIMENT_NAME=stage2-rl-level1to2-qwen2.5-7b-instruct-nb-v2-easier-only-nb-v2-easier-split
export TRAIN_FILES="['data/numerical_task/stage2_rl_level1_inverse_nb_v2_easier/train.parquet']"
export VAL_FILES="['data/numerical_task/stage2_rl_level1_inverse_nb_v2_easier/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh

