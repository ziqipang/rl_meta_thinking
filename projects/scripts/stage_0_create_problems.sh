python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task_v2/stage1_level1/train.parquet \
    --stage 1 \
    --split train \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task_v2/stage1_level1/test.parquet \
    --stage 1 \
    --split test \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024