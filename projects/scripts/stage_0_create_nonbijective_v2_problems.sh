python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage1_level1_nb_v2/train.parquet \
    --stage 1 \
    --split train \
    --function_kind nonbijective_v2 \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage1_level1_nb_v2/test.parquet \
    --stage 1 \
    --split test \
    --function_kind nonbijective_v2 \
    --stage1_pool all \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024
