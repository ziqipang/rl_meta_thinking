export PROJECT_NAME=numerical-task
export EXPERIMENT_NAME=stage1-rft-qwen2.5-7b-instruct-old
export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=data/numerical_task/stage_1_sft_data/train.parquet 
export VAL_FILES=data/numerical_task/stage_1_sft_data/test.parquet 
export BATCH_SIZE=128
export EPOCHS=2
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh