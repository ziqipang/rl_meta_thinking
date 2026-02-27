mkdir -p data/numerical_task/diverse_evals/lenchange_functions

python3 projects/datasets/diverse_evals/create_lenchange_multi_preimage_eval.py \
  --save_path data/numerical_task/diverse_evals/lenchange_functions/test.parquet \
  --data_num 1024 \
  --output_min_len 6 \
  --output_max_len 10 \
  --num_answers_min 3 \
  --num_answers_max 5

