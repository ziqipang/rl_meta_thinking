set -x

export MODEL_PATH=disk/checkpoints/numerical-task/stage1-rft-qwen2.5-7b-instruct-nb-v2
export PROJECT_NAME=numerical-task-rl
export EXPERIMENT_NAME=stage2-rl-level1to2-qwen2.5-7b-instruct-multi-response-with-none
export TRAIN_FILES="['data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/train.parquet','data/numerical_task_v2/multi_response_with_none/bijective/train.parquet']"
export VAL_FILES="['data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/test.parquet','data/numerical_task_v2/multi_response_with_none/bijective/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_more_candidates/test.parquet','data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs_more_candidates/test.parquet','data/numerical_task_v2/diverse_evals/lenchange_functions/test.parquet','data/numerical_task_v2/stage2_rl_level1_inverse/test.parquet']"
export NNODES=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

export TOKENIZERS_PARALLELISM=true
export RAY_DEBUG=legacy

# Multi-preimage rewards can be sparse early; disabling filter_groups avoids dropping all prompts.
# When disabled, verl requires `data.gen_batch_size == data.train_batch_size`.
bash examples/grpo_trainer/template.sh
