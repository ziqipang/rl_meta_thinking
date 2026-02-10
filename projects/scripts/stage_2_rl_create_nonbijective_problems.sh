python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1-2_nb/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 2 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level2-2_nb/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective \
    --min_level 2 \
    --max_level 2 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1-2_nb/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 4 \
    --data_num 2048
