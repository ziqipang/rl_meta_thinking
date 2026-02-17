set -x

export NB_V2_SFT_DATA_SAVE_PATH=data/numerical_task/stage1_level1_nb_v2/rft_data

# Step 1: Build RFT data from rollouts
mkdir -p ${NB_V2_SFT_DATA_SAVE_PATH}
python3 projects/datasets/rft_data.py \
    --gen_path data/numerical_task/stage1_level1_nb_v2/rollout.parquet \
    --data_path data/numerical_task/stage1_level1_nb_v2/train.parquet \
    --save_path ${NB_V2_SFT_DATA_SAVE_PATH} \
    --val_size 1024

# Step 2: Train SFT (RFT) with Qwen2.5-3B-Instruct
export PROJECT_NAME=numerical-task
export EXPERIMENT_NAME=stage1-rft-qwen2.5-3b-instruct-nb-v2
export MODEL_PATH=disk/models/Qwen2.5-3B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=${NB_V2_SFT_DATA_SAVE_PATH}/train.parquet
export VAL_FILES=${NB_V2_SFT_DATA_SAVE_PATH}/test.parquet
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
