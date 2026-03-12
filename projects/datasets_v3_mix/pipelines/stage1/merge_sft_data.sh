#!/usr/bin/env bash
# Subset each RFT dataset to 1/3, then merge into a combined SFT dataset.
set -euo pipefail
set -x

BIJECTIVE_V3_RFT=data/numerical_task_v3/stage1_level1/rft_data
NB_V2_RFT=data/numerical_task_v2/stage1_level1_nb_v2/rft_data
SUBSET_DIR=data/numerical_task_v3_mix/stage_1_sft_subsets
COMBINED_SFT_DATA=data/numerical_task_v3_mix/stage_1_sft_data_combined

SUBSET_FRACTION="1/3"

mkdir -p "${SUBSET_DIR}"

# Subset each RFT train set to 1/3 and merge with test sets
python3 -c "
import pandas as pd, os

frac = ${SUBSET_FRACTION}
subset_dir = '${SUBSET_DIR}'
combined_dir = '${COMBINED_SFT_DATA}'
os.makedirs(combined_dir, exist_ok=True)

bij_train = pd.read_parquet('${BIJECTIVE_V3_RFT}/train.parquet')
nb_train = pd.read_parquet('${NB_V2_RFT}/train.parquet')
bij_test = pd.read_parquet('${BIJECTIVE_V3_RFT}/test.parquet')
nb_test = pd.read_parquet('${NB_V2_RFT}/test.parquet')

bij_n = max(1, int(len(bij_train) * frac))
nb_n = max(1, int(len(nb_train) * frac))

print(f'V3 bijective train: {len(bij_train)} -> {bij_n} (1/3)')
print(f'V2 nb train: {len(nb_train)} -> {nb_n} (1/3)')

bij_sub = bij_train.sample(n=bij_n, random_state=42).reset_index(drop=True)
nb_sub = nb_train.sample(n=nb_n, random_state=42).reset_index(drop=True)

bij_sub.to_parquet(os.path.join(subset_dir, 'bijective_v3_train_1of3.parquet'), index=False)
nb_sub.to_parquet(os.path.join(subset_dir, 'nb_v2_train_1of3.parquet'), index=False)

merged_train = pd.concat([bij_sub, nb_sub], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
merged_test = pd.concat([bij_test, nb_test], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

merged_train.to_parquet(os.path.join(combined_dir, 'train.parquet'), index=False)
merged_test.to_parquet(os.path.join(combined_dir, 'test.parquet'), index=False)

print(f'Combined train: {len(merged_train)} rows -> {combined_dir}/train.parquet')
print(f'Combined test: {len(merged_test)} rows -> {combined_dir}/test.parquet')
"
