# Multi Response With None

This folder contains the dataset creation scripts and Stage 2 RL launch scripts
for the shared multi-response inverse setting:

- bijective inverse tasks use the same multi-answer output format as non-bijective tasks
- the prompt allows fewer than `num_answers` outputs when fewer valid answers exist
- the mixed run trains on both bijective and non-bijective data together

Run all commands from the repo root:

```bash
cd /home/ubuntu/rl_meta_thinking
```

## 1. Create the datasets

Generate both the bijective and non-bijective datasets:

```bash
bash projects/scripts/multi_response_with_none/create_all_problems.sh
```

This writes:

- `data/numerical_task_v2/multi_response_with_none/bijective/train.parquet`
- `data/numerical_task_v2/multi_response_with_none/bijective/test.parquet`
- `data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/train.parquet`
- `data/numerical_task_v2/multi_response_with_none/nonbijective_v2_balanced/test.parquet`

If you only want one family, run one of:

```bash
bash projects/scripts/multi_response_with_none/create_bijective_problems.sh
bash projects/scripts/multi_response_with_none/create_nonbijective_problems.sh
```

Notes:

- These scripts currently generate level-1 Stage 2 data.
- `num_answers` is currently set to `2` in both creation scripts.
- The non-bijective data uses `nonbijective_v2` with the `balanced` split.

## 2. Launch Stage 2 RL

Default mixed run:

```bash
bash projects/scripts/multi_response_with_none/stage_2_rl_multi_inverse_nb.sh
```

This trains on:

- bijective multi-response inverse data
- non-bijective multi-response inverse data

Single-family ablations:

```bash
bash projects/scripts/multi_response_with_none/stage_2_rl_bijective_only.sh
bash projects/scripts/multi_response_with_none/stage_2_rl_nb_only.sh
```

## 3. Check the default paths before launching

All three Stage 2 RL scripts currently assume:

- `MODEL_PATH=disk/checkpoints/numerical-task/stage1-rft-qwen2.5-7b-instruct-nb-v2`
- `PROJECT_NAME=numerical-task-rl`
- outputs are saved under `disk/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}`

If your Stage 1 checkpoint lives elsewhere, edit `MODEL_PATH` in the launch
script before running it.

## 4. Which script should I use?

Use:

- `stage_2_rl_multi_inverse_nb.sh` for the main merged bijective + non-bijective experiment
- `stage_2_rl_bijective_only.sh` for a bijective-only ablation
- `stage_2_rl_nb_only.sh` for a non-bijective-only ablation

## 5. Minimal launch sequence

```bash
cd /home/ubuntu/rl_meta_thinking
bash projects/scripts/multi_response_with_none/create_all_problems.sh
bash projects/scripts/multi_response_with_none/stage_2_rl_multi_inverse_nb.sh
```
