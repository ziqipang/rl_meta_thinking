mkdir -p data/numerical_task/diverse_evals/nb_v2_balanced_more_candidates

python3 projects/datasets/diverse_evals/create_multi_preimage_eval.py \
  --save_path data/numerical_task/diverse_evals/nb_v2_balanced_more_candidates/test.parquet \
  --stage 2 \
  --split test \
  --function_kind nonbijective_v2 \
  --nb_v2_split balanced \
  --min_level 1 \
  --max_level 1 \
  --data_num 1024 \
  --num_answers_min 3 \
  --num_answers_max 6 \
  --min_len 2 \
  --max_len 6 \
  --search_attempts 8000 \
  --data_source_tag eval-more-candidates

