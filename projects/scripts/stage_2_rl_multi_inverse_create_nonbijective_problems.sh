mkdir -p data/numerical_task/stage2_rl_level1_multi_preimage_nb_v2_balanced

python3 projects/datasets/create_string_data_multi.py \
    --save_path data/numerical_task/stage2_rl_level1_multi_preimage_nb_v2_balanced/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets/create_string_data_multi.py \
    --save_path data/numerical_task/stage2_rl_level1_multi_preimage_nb_v2_balanced/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2
