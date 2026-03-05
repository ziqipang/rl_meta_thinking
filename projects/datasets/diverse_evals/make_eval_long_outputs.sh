mkdir -p data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs

python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
  --save_path data/numerical_task_v2/diverse_evals/nb_v2_balanced_long_outputs/test.parquet \
  --stage 2 \
  --split test \
  --function_kind nonbijective_v2 \
  --nb_v2_split balanced \
  --min_level 1 \
  --max_level 1 \
  --data_num 1024 \
  --num_answers 2 \
  --min_len 6 \
  --max_len 10 \
  --output_min_len 6 \
  --output_max_len 10 \
  --search_attempts 20000 \
  --data_source_tag eval-long-outputs

