# Datasets

This directory contains small dataset-generation and dataset-processing utilities used by the training scripts in this repo.

## `create_string_data.py`

Generates CodeIO-style examples where a digit-string input is transformed by a composed function expression (e.g. `func_03(func_10(x))`).

### What it generates

Each sample is a JSON object with:
- `prompt`: a chat-style prompt containing a Python snippet and either an input (forward) or an output (backward)
- `reward_model.ground_truth`: JSON with `ref_input`, `ref_output`, and `ref_code`

Output formats:
- `.jsonl`: always supported
- `.parquet`: supported if the optional `datasets` dependency (and a parquet backend) is installed

### Function families

`create_string_data.py` defines:
- Bijective functions: `func_00`..`func_23` (unique inverse)
- Non-bijective functions: `func_24`..`func_43` (many-to-one)

Non-bijective families included:
- Family A (digitwise collapses): `func_24`..`func_33`
- Family B (positional masking): `func_34`..`func_36`
- Family C (pairwise aggregation): `func_37`..`func_39`
- Family D (sorting): `func_40`..`func_42`
- Family E (neighbor-coupled): `func_43`

### Train/test function pools

By default, the script uses separate train/test pools:
- Bijective train: `func_00..func_17`, bijective test: `func_18..func_23`
- Non-bijective train: `func_24..func_33`, non-bijective test: `func_34..func_43`

### CLI highlights

Required:
- `--save_path PATH`
- `--stage {1,2}`
- `--split {train,test}`

Common options:
- `--direction {forward,backward}` (default: `forward`)
- `--min_level L`, `--max_level L` (composition depth range)
- `--data_num N` (must be divisible by number of levels)

Function selection:
- `--function_kind {bijective,nonbijective,both}` (default: `bijective`)
- `--stage1_pool {all,split}`
  - `all`: for `--stage 1`, sample from all selected functions (ignores train/test pools)
  - `split`: for `--stage 1`, sample from the train/test pool for the chosen `--split`

Notes:
- `--direction backward` is only supported when the selected pool is bijective (otherwise the inverse is not unique).

### Examples

Reproduce the default bijective stage-2 train split (matches `projects/scripts/stage_2_rl_create_problems.sh`):
```bash
python3 projects/datasets/create_string_data.py \
  --save_path data/numerical_task/stage2_rl_level1-2/train.parquet \
  --stage 2 --split train --min_level 1 --max_level 2 --data_num 50000
```

Generate non-bijective forward-only training data from Family A only:
```bash
python3 projects/datasets/create_string_data.py \
  --save_path /tmp/nonbij_train.jsonl \
  --stage 1 --split train --function_kind nonbijective --stage1_pool split \
  --min_level 1 --max_level 2 --data_num 2000
```

Generate a mixed pool (bijective + non-bijective) for forward direction:
```bash
python3 projects/datasets/create_string_data.py \
  --save_path /tmp/mixed_forward.jsonl \
  --stage 1 --split train --function_kind both --stage1_pool split \
  --min_level 1 --max_level 2 --data_num 2000
```

## `rft_data.py`

Builds an SFT dataset from rollout generations by keeping only correct responses (as judged by `verl.utils.reward_score.codeio.compute_score`), optionally filtering by length and formatting.

Usage (example):
```bash
python3 projects/datasets/rft_data.py \
  --gen_path data/numerical_task/stage1_level1/rollout.parquet \
  --data_path data/numerical_task/stage1_level1/train.parquet \
  --save_path data/numerical_task/stage_1_sft_data \
  --val_size 1024
```

Notes:
- The script currently contains an interactive breakpoint (`pdb.set_trace()`), so it will pause unless you remove/skip it.
- Requires `datasets`, and may require `transformers` if you use `--max_length` / `--tokenizer`.
