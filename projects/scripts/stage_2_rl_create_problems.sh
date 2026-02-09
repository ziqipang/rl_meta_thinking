python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1-2/train.parquet \
    --stage 2 \
    --split train \
    --min_level 1 \
    --max_level 2 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level2-2/train.parquet \
    --stage 2 \
    --split train \
    --min_level 2 \
    --max_level 2 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1-2/test.parquet \
    --stage 2 \
    --split test \
    --min_level 1 \
    --max_level 4 \
    --data_num 2048