#!/usr/bin/env bash
# Build RFT (rejection fine-tuning) data from bijective rollouts.
set -euo pipefail
set -x

DATA_DIR=data/stage1_bijective
SAVE_DIR=${DATA_DIR}/rft_data

mkdir -p "${SAVE_DIR}"
python3 projects/tools/rft_data.py \
    --gen_path ${DATA_DIR}/rollout.parquet \
    --data_path ${DATA_DIR}/train.parquet \
    --save_path "${SAVE_DIR}" \
    --val_size 1024
