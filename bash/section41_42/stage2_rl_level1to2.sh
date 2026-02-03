set -x

export MODEL_PATH=disk/checkpoints/string-task/stage1-rft-llama-3.1-8b-instruct-use-downloaded-data
export PROJECT_NAME=string-task
export EXPERIMENT_NAME=stage2-rl-level1to2-llama-own-rft-downloaded-data
export TRAIN_FILES="['data/string_task/stage2_level1/train.parquet','data/string_task/stage2_level2/train.parquet']"
export VAL_FILES="['data/string_task/stage2_level1to8/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

bash examples/grpo_trainer/template.sh
