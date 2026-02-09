python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse/train.parquet \
    --stage 2 \
    --split train \
    --direction backward \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse/test.parquet \
    --stage 2 \
    --split test \
    --direction backward \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048