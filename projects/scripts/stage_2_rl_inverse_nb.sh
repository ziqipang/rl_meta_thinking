set -x

export MODEL_PATH=disk/checkpoints/numerical-task/stage1-rft-qwen2.5-7b-instruct-nb-v2
export PROJECT_NAME=numerical-task-rl
export EXPERIMENT_NAME=stage2-rl-level1to2-qwen2.5-7b-instruct-nb-v2-easier
export TRAIN_FILES="['data/numerical_task/stage2_rl_level1_inverse_nb_v2/train.parquet','data/numerical_task/stage2_rl_level1_inverse/train.parquet']"
export VAL_FILES="['data/numerical_task/stage2_rl_level1_inverse_nb_v2/test.parquet','data/numerical_task/stage2_rl_level1_inverse/test.parquet','data/numerical_task/stage2_rl_level1-2/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

# NOTE: For sparse/binary rewards (e.g. inverse/preimage), filter_groups can discard all prompts
# early on (std==0 across n rollouts), causing:
#   ValueError: num_gen_batches >= max_num_gen_batches. Generated too many.
# Disable it for this run.
#
# When filter_groups is disabled, verl requires `data.gen_batch_size == data.train_batch_size`.
# The GRPO template defaults gen_batch_size to `4 * train_batch_size`, so override it here.
# Use OmegaConf interpolation so this stays correct if you override train_batch_size via CLI.
bash examples/grpo_trainer/template.sh
# bash examples/grpo_trainer/template.sh "$@" algorithm.filter_groups.enable=False data.gen_batch_size='${data.train_batch_size}'
