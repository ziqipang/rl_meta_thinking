set -x

export NB_SFT_DATA_SAVE_PATH=data/numerical_task/stage1_level1_nb_v2/rft_data
export COMBINED_SFT_DATA_SAVE_PATH=data/numerical_task/stage_1_sft_data_nb_v2_combined

# RFT data creation is assumed done already (annotated). This script only trains.
# Merge bijective+nb SFT parquets into a single folder to avoid multi-file config parsing issues.
python3 projects/datasets/merge_sft_parquets.py \
    --out_dir ${COMBINED_SFT_DATA_SAVE_PATH} \
    --train_files data/numerical_task/stage_1_sft_data/train.parquet ${NB_SFT_DATA_SAVE_PATH}/train.parquet \
    --val_files data/numerical_task/stage_1_sft_data/test.parquet ${NB_SFT_DATA_SAVE_PATH}/test.parquet

export PROJECT_NAME=numerical-task
export EXPERIMENT_NAME=stage1-rft-qwen2.5-3b-instruct-nb-v2
export MODEL_PATH=disk/models/Qwen2.5-7B-Instruct
export NNODES=1
export SP_SIZE=1
export MAX_LENGTH=5120
export TRAIN_FILES=${COMBINED_SFT_DATA_SAVE_PATH}/train.parquet
export VAL_FILES=${COMBINED_SFT_DATA_SAVE_PATH}/test.parquet
export BATCH_SIZE=128
export EPOCHS=1
export SAVE_DIR=disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

bash examples/sft/template.sh
