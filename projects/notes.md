# Procedures / Script Map (quick)

## Where scripts live
- Dataset creation + training launchers: `projects/scripts/`
- Example RL launchers I run from repo root: `bash/section41_42/`
- Dataset generator: `projects/datasets/create_string_data.py`

## Stage 0/1: SFT (RFT-style)
- Create raw stage-1 problems (bijective): `projects/scripts/stage_0_create_problems.sh` → writes `data/numerical_task/stage1_level*/{train,test}.parquet`
- Create raw stage-1 problems (non-bijective): `projects/scripts/stage_0_create_nonbijective_problems.sh` → writes `data/numerical_task/stage1_level*_nb/{train,test}.parquet`
- Generate model rollouts (for filtering into SFT):
  - bijective: `projects/scripts/stage_0_rejection_sampling.sh` → `.../rollout.parquet`
  - non-bijective: `projects/scripts/stage_0_rejection_sampling_nb.sh` → `..._nb/rollout.parquet`
- Build SFT data from rollouts: `python3 projects/datasets/rft_data.py --gen_path .../rollout.parquet --data_path .../train.parquet --save_path ...`
- Train SFT:
  - bijective: `projects/scripts/stage_1_rft.sh`
  - mixed (bijective+nb merged): `projects/scripts/stage_1_rft_with_nb.sh`

## Stage 2: RL data creation
- Forward (bijective): `projects/scripts/stage_2_rl_create_problems.sh` → `data/numerical_task/stage2_rl_level*/...parquet`
- Forward (non-bijective): `projects/scripts/stage_2_rl_create_nonbijective_problems.sh` → `data/numerical_task/stage2_rl_level*_nb/...parquet`
- “Inverse” style:
  - Bijective unique-inverse: `projects/scripts/stage_2_rl_inverse_create_problems.sh` (`--direction backward`) → `.../stage2_rl_level1_inverse/...parquet`
  - Non-bijective preimage (find any x): `projects/scripts/stage_2_rl_inverse_create_nonbijective_problems.sh` (`--direction backward --function_kind nonbijective`) → `.../stage2_rl_level1_inverse_nb/...parquet`

## Stage 2: RL training hookup
- GRPO template runner: `projects/scripts/stage_2_rl_grpo.sh` (expects `TRAIN_FILES` / `VAL_FILES` env vars pointing at the stage-2 parquet(s))
- Example launchers in `bash/section41_42/` typically set:
  - `MODEL_PATH` (from stage-1 SFT checkpoint)
  - `TRAIN_FILES` / `VAL_FILES` (from stage-2 creation scripts)
  - then call `bash examples/grpo_trainer/template.sh ...`

## Logging / reward notes
- `create_string_data.py` sets `data_source`:
  - bijective: `codeio-*`
  - non-bijective: `codeio-nb-*`
  - non-bijective backward uses `codeio-nb-preimage-*` (not unique inverse).
- For `codeio-nb-preimage-*`, reward scoring executes `ground_truth.ref_code` with `func_XX` loaded from `projects/datasets/create_string_data.py` (so stage-2 “main_solution-only” code still runs).
