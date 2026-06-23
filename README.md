# RL Meta-Thinking on Bijective + Non-Bijective Function Tasks

Minimal pipeline to reproduce the SFT + RL experiments on digit-string function
inversion. The setup co-trains a Qwen2.5-7B-Instruct model on two function
families (24 bijective + 25 non-bijective) and then runs RL with the function
implementations *hidden* from the prompt, so the model has to recall them from
the SFT phase.

The two RL experiments that this repo is set up for:

- **Non-bijective only** — RL fine-tune on non-bijective preimage tasks alone.
- **Co-training** — RL fine-tune on bijective + non-bijective tasks together.

Training engine: a trimmed copy of [veRL](https://github.com/volcengine/verl)
under `verl/`. All custom data generation lives under `projects/`.

---

## Environment setup

1. Clone the repository and create a Python environment (Python 3.10+ tested).

   ```bash
   git clone <repo-url> rl_meta_thinking
   cd rl_meta_thinking
   virtualenv .venv && source .venv/bin/activate
   pip install -e .
   ```

2. Pin the two libraries that the pipeline expects:

   ```bash
   pip install "transformers==4.43"
   pip install "flash-attn==2.7.4.post1" --no-build-isolation
   ```

3. Authenticate with Weights & Biases (training jobs log there):

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
│   ├── create_bijective.py                 # bijective stage-1 / stage-2 data
│   ├── create_bijective_multi_response.py  # bijective stage-2 multi-response
│   ├── create_nonbijective.py              # non-bijective func library + stage-1 data
│   ├── create_nonbijective_multi.py        # non-bijective multi-preimage helpers
│   ├── create_multi_response_with_none.py  # non-bijective stage-2 multi-response
│   └── diverse_evals/                      # extra evaluation slices
├── pipelines/                   # one-shot bash launchers
│   ├── stage0/                  # rejection-sampling SFT data
│   ├── stage1/                  # SFT training
│   └── stage2/                  # RL training (this is the headline experiment)
└── tools/
    ├── rft_data.py                          # filter rollouts into SFT data
    └── merge_latest_sft_checkpoint.py       # promote latest SFT step to a stable dir

verl/                            # training engine (vendored)
examples/grpo_trainer/template.sh # GRPO entrypoint used by stage 2
examples/sft/template.sh         # SFT entrypoint used by stage 1
scripts/model_merger.py          # post-RL FSDP shard merger
```

---

## Pipeline

### Stage 0 — rejection-sampling SFT data

Generate stage-1 (with-code) problems for both function families, sample 10
rollouts per prompt from Qwen2.5-7B-Instruct, and filter into RFT splits.

```bash
bash projects/pipelines/stage0/run_rejection_sampling_pipeline.sh
```

Output:
- `data/stage1_bijective/rft_data/{train,test}.parquet`
- `data/stage1_nonbijective/rft_data/{train,test}.parquet`

You can also run the bijective and non-bijective halves individually
(`create_*_problems.sh` → `rejection_sampling_*.sh` → `rft_*.sh`).

### Stage 1 — SFT (RFT) on the combined dataset

Merge the two RFT splits (1/3 of each), train SFT, then promote the latest
checkpoint.

```bash
bash projects/pipelines/stage1/merge_sft_data.sh
bash projects/pipelines/stage1/train_sft.sh
bash projects/pipelines/stage1/merge_sft_model.sh
```

Output checkpoint: `disk/models/reasoning-data/stage1-rft-qwen2.5-7b-instruct/`

### Stage 2 — RL

Generate stage-2 RL training/eval data (prompts contain *no* function code; the
model must rely on the function semantics it learned during SFT).

```bash
bash projects/pipelines/stage2/prepare_data.sh
```

Output:
- `data/stage2/{bijective,nonbijective}/{train,test}.parquet`
- `data/stage2/bijective_inverse/{train,test}.parquet`
- `data/eval/{long_outputs,more_candidates,long_outputs_more_candidates,lenchange_functions}/test.parquet`

Then launch one of the two RL experiments:

```bash
# RL on non-bijective only
bash projects/pipelines/stage2/rl_nonbijective_only.sh

# RL co-training on bijective + non-bijective
bash projects/pipelines/stage2/rl_both_funcs.sh
```

Each script runs 160 GRPO steps on 8 GPUs by default; tune `NNODES`,
`MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, or `SAVE_DIR` to relocate.

After training finishes, merge FSDP shards back into a HuggingFace-format
checkpoint:

```bash
python3 scripts/model_merger.py \
    --local_dir disk/checkpoints/reasoning-data-rl/<experiment_name>/global_step_<N>/actor
```
