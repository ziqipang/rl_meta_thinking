mkdir -p data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced

python3 projects/datasets/create_multi_response_with_none.py \
    --save_path data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/train.parquet \
    --stage 2 \
    --split train \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 50000 \
    --num_answers 2

python3 projects/datasets/create_multi_response_with_none.py \
    --save_path data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/test.parquet \
    --stage 2 \
    --split test \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 1024 \
    --num_answers 2
