export NNODES=1
export DATA_PATH=data/numerical_task/stage2_rl_level1_inverse_easy_asf_same_set/test.parquet
export N_SAMPLES=4
export SAVE_PATH=data/numerical_task/stage2_rl_level1_inverse_easy_asf_same_set/rollout.parquet
# export MODEL_PATH=disk/checkpoints/numerical-task-rl/stage2-rl-level1-inverse-easy-asf-qwen2.5-7b-instruct-fix-data/global_step_20/actor/huggingface
export MODEL_PATH=disk/checkpoints/numerical-task/stage1-rft-qwen2.5-7b-instruct-old
export TEMPERATURE=1.0
export PROMPT_LENGTH=1024
export RESPONSE_LENGTH=8192

bash examples/generation/run_string.sh