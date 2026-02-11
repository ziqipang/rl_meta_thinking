python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse_nb/train.parquet \
    --stage 2 \
    --split train \
    --direction backward \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse_nb/test.parquet \
    --stage 2 \
    --split test \
    --direction backward \
    --function_kind nonbijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 2048
