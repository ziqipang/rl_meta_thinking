# RL on Bijective + Non-Bijective Function Tasks

Minimal pipeline to run GRPO reinforcement learning on digit-string function
inversion tasks, starting directly from Qwen2.5-7B-Instruct.

The function code is included in every prompt, so no SFT warm-up is needed —
GRPO runs on top of the base instruct model.

Two RL experiments are set up:

- **Non-bijective only** — train on non-bijective preimage tasks alone.
- **Co-training** — train on bijective + non-bijective tasks together.

The training engine is a trimmed copy of [veRL](https://github.com/volcengine/verl)
under `verl/`. All custom data generation lives under `projects/`.

---

## Environment setup

1. Clone and create a Python environment (Python 3.10+ tested).

   ```bash
   git clone <repo-url> rl_meta_thinking
   cd rl_meta_thinking
   virtualenv .venv && source .venv/bin/activate
   pip install -e .
   ```

2. Pin the two libraries the pipeline expects:

   ```bash
   pip install "transformers==4.43"
   pip install "flash-attn==2.7.4.post1" --no-build-isolation
   ```

3. Authenticate with Weights & Biases:

   ```bash
   export WANDB_API_KEY="..."
   ```

4. Download the base model into the path the pipeline expects:

   ```bash
   mkdir -p disk/models
   hf download --local-dir disk/models/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct
   ```

   (Or symlink your existing checkpoint to `disk/models/Qwen2.5-7B-Instruct`.)

---

## Repository layout

```
projects/
├── atomic_functions.py          # bijective func library (func_00..23)
├── generators/                  # data generators
│   ├── create_bijective.py                 # bijective forward / inverse data
│   ├── create_bijective_multi_response.py  # bijective multi-response
│   ├── create_nonbijective.py              # non-bijective func library + base generator
│   ├── create_nonbijective_multi.py        # non-bijective multi-preimage helpers
│   ├── create_multi_response_with_none.py  # non-bijective multi-response
│   └── diverse_evals/                      # extra evaluation slices
└── pipelines/
    ├── prepare_data.sh                     # generate RL train + eval data
    ├── rl_nonbijective_only.sh             # GRPO on non-bijective only
    └── rl_both_funcs.sh                    # GRPO co-training on both families

verl/                            # training engine (vendored)
examples/grpo_trainer/template.sh # GRPO entrypoint used by RL scripts
scripts/model_merger.py          # post-RL FSDP shard merger
```

---

## Run the pipeline

### Step 1 — generate RL data

```bash
bash projects/pipelines/prepare_data.sh
```

Outputs:
- `data/stage2/{bijective,nonbijective}/{train,test}.parquet` — training + eval
- `data/stage2/bijective_inverse/test.parquet` — bijective inverse eval
- `data/eval/{long_outputs,more_candidates,long_outputs_more_candidates,lenchange_functions}/test.parquet` — diverse non-bijective evals

### Step 2 — launch GRPO

Each script runs 200 GRPO steps on 8 GPUs by default. Tune `NNODES`,
`MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, or `SAVE_DIR` in the script to relocate.

```bash
# RL on non-bijective only
bash projects/pipelines/rl_nonbijective_only.sh

# RL co-training on bijective + non-bijective
bash projects/pipelines/rl_both_funcs.sh
```

### Step 3 — merge the FSDP checkpoint (after training)

```bash
python3 scripts/model_merger.py \
    --local_dir disk/checkpoints/reasoning-data-rl/<experiment_name>/global_step_<N>/actor
```
