# datasets_v3_mix

Mixed pipeline combining **v3 bijective functions** with **v2 non-bijective functions**, using the v2-style SFT-then-RL approach (no code in prompt during RL).

## Design

Unlike `datasets_v3/pipelines/exp_code_in_prompt/` which provides function code in the RL prompt and skips SFT, this pipeline:

1. **SFT phase** — trains the model on forward problems with full code (stage 1), so it memorizes both function families.
2. **RL phase** — presents only `def main_solution(x): return func_XX(x)` (stage 2, no code), forcing the model to rely on memorized function semantics.

This design more effectively reveals spurious reasoning patterns from RL, because the model cannot verify its reasoning by re-reading the function implementation.

### Function naming

- V3 bijective: `func_00`..`func_23` (from `projects/datasets_v3/atomic_functions.py`)
- V2 non-bijective: `nb_00`..`nb_24` → displayed as `func_24`..`func_48` in prompts (via `func_name_mapping` in `projects/datasets/create_string_data.py`)

No naming collision between the two families.

### Reward scoring

- V3 bijective multi-response uses `ref_code` with stage=1 (full source code), so the scorer can execute it directly without needing v3 functions in its environment.
- V2 non-bijective multi-response uses `ref_code` with stage=2 (only `main_solution`), and the scorer resolves function names via `_CODEIO_FUNC_ENV`.

## Pipeline

### Stage 0: Rejection sampling (v3 bijective)

```bash
# Full pipeline (create problems → rollout → RFT data)
bash projects/datasets_v3_mix/pipelines/stage0/run_rejection_sampling_pipeline.sh
```

Existing data:
- V3 bijective RFT: `data/numerical_task_v3/stage1_level1/rft_data/`
- V2 non-bijective RFT: `data/numerical_task_v2/stage1_level1_nb_v2/rft_data/`

### Stage 1: SFT

```bash
# Merge RFT datasets + train SFT
bash projects/datasets_v3_mix/pipelines/stage1/merge_sft_data.sh
bash projects/datasets_v3_mix/pipelines/stage1/train_sft.sh
bash projects/datasets_v3_mix/pipelines/stage1/merge_sft_model.sh
```

### Stage 2: RL

```bash
# Generate RL training + eval data
bash projects/datasets_v3_mix/pipelines/stage2/prepare_data.sh

# Experiment 1: co-train bijective + non-bijective
bash projects/datasets_v3_mix/pipelines/stage2/rl_both_funcs.sh

# Experiment 2: non-bijective only
bash projects/datasets_v3_mix/pipelines/stage2/rl_nonbijective_only.sh
```

## Data layout

```
data/numerical_task_v3_mix/
├── stage_1_sft_data_combined/       # Merged SFT data
│   ├── train.parquet
│   └── test.parquet
├── multi_response_with_none/
│   ├── bijective_v3/                # V3 bijective (stage-2 prompt, stage-1 ref_code)
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── nonbijective_v2_balanced/    # V2 non-bijective (stage-2 prompt)
│       ├── train.parquet
│       └── test.parquet
└── stage2_rl_level1_inverse/        # V3 bijective inverse (eval)
    ├── train.parquet
    └── test.parquet
```
