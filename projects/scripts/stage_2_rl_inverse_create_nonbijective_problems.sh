python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse_nb_v2_original/train.parquet \
    --stage 2 \
    --split train \
    --direction backward \
    --function_kind nonbijective_v2 \
    --nb_v2_split original \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000

python3 projects/datasets/create_string_data.py \
    --save_path data/numerical_task/stage2_rl_level1_inverse_nb_v2_original/test.parquet \
    --stage 2 \
    --split test \
    --direction backward \
    --function_kind nonbijective_v2 \
    --nb_v2_split original \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024
