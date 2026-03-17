#!/usr/bin/env bash
# Generate forward function evaluation data for the "code-in-prompt" experiments.
# Forward = given input x, predict f(x).
# Uses ONLY train-set functions (not val/test) for both bijective and non-bijective.
#
# We generate enough forward data for two experiments:
#   - 10% forward: 5,556 bij + 5,556 nb = 11,112 total forward
#   - 20% forward: 12,500 bij + 12,500 nb = 25,000 total forward
#
# We generate the full 20% amounts and then downsample for the 10% experiment.
set -euo pipefail
set -x

cd /home/ubuntu/rl_meta_thinking

DATA_ROOT=data/numerical_task_v3_mix_code_in_prompt

# ============================================================
# 1. Bijective forward (v3 generator, train-set functions only)
# ============================================================
mkdir -p ${DATA_ROOT}/forward/bijective_v3

echo "=== Generating v3 bijective forward (code in prompt, train-set only) ==="
python3 projects/datasets_v3/generators/create_string_data.py \
    --save_path ${DATA_ROOT}/forward/bijective_v3/train_12500.parquet \
    --stage 1 \
    --split train \
    --stage1_pool split \
    --direction forward \
    --function_kind bijective \
    --min_level 1 \
    --max_level 1 \
    --data_num 12500

# ============================================================
# 2. Non-bijective forward (v2 generator, train-set functions only)
# ============================================================
mkdir -p ${DATA_ROOT}/forward/nonbijective_v2_balanced

echo "=== Generating v2 non-bijective forward (code in prompt, train-set only) ==="
python3 projects/datasets/create_string_data.py \
    --save_path ${DATA_ROOT}/forward/nonbijective_v2_balanced/train_12500.parquet \
    --stage 1 \
    --stage1_pool split \
    --split train \
    --direction forward \
    --function_kind nonbijective_v2 \
    --nb_v2_split balanced \
    --min_level 1 \
    --max_level 1 \
    --data_num 12500

# ============================================================
# 3. Downsample to create 10% forward datasets (5,556 each)
# ============================================================
echo "=== Downsampling forward data for 10% experiment ==="
python3 -c "
import pandas as pd

for kind, path in [
    ('bijective_v3', '${DATA_ROOT}/forward/bijective_v3'),
    ('nonbijective_v2_balanced', '${DATA_ROOT}/forward/nonbijective_v2_balanced'),
]:
    df = pd.read_parquet(f'{path}/train_12500.parquet')
    print(f'{kind}: {len(df)} rows, data_sources: {df[\"data_source\"].unique()}')
    ds = df.sample(n=5556, random_state=42).reset_index(drop=True)
    ds.to_parquet(f'{path}/train_5556.parquet')
    print(f'  Downsampled to {len(ds)} rows -> {path}/train_5556.parquet')
"

echo "=== Forward data preparation complete ==="
echo "Data root: ${DATA_ROOT}/forward"
find ${DATA_ROOT}/forward -name "*.parquet" -exec ls -lh {} \;
