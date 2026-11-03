# Not All Reasoning Generalizes: A Case Study on Learning Generalizable Reasoning Strategies in RLVR

Ziqi Pang, Yu-Xiong Wang

University of Illinois Urbana-Champaign

[[Paper (arXiv)]](https://arxiv.org/abs/2611.03627)

---

## TL;DR

Reinforcement learning with verifiable rewards (RLVR) can teach an LLM the *right*
reasoning strategy in-distribution, yet the same model abandons it under
out-of-distribution (OOD) shifts. We study this with a controllable
equation-solving task, finding multiple inputs `x` such that `f(x) = y` for a
non-bijective string function `f`, and show:

1. **Reasoning-strategy collapse.** RLVR on non-bijective inversion (**Mono**)
   learns structured, step-by-step *derivation* and reaches high in-distribution
   accuracy. On harder OOD tasks (more solutions, longer `y`, length-changing
   functions), it falls back to brute-force *enumeration*, and accuracy drops sharply.
   The cause is neither an incapable base model nor reward hacking.
2. **An indirect fix.** Co-training with **bijective** inversion tasks (**Mixed**)
   roughly doubles the use of derivation on OOD tasks and substantially improves OOD
   accuracy. This happens even though bijective tasks ask for a single answer, are not
   uniformly harder, and sit further from the multi-solution OOD evaluation.
3. **A bits-per-token explanation.** Viewing reasoning as shrinking the candidate
   solution space, derivation yields more bits of information per token than
   enumeration. We show that outcome-only RLVR (actor-critic and GRPO) implicitly credits
   each token by this progress, so tasks that sharpen the bit-rate gap, such as bijective
   tasks, a shorter token budget, or a constrained solution space, favor derivation.

## Framework

| | Setting |
|---|---|
| **Task** | Given the Python source of `f` and a target `y`, output `k` distinct digit strings `x` with `f(x) = y` |
| **Functions** | 24 non-bijective functions (18 train / 6 held-out), plus 24 bijective functions for co-training |
| **In-distribution** | Held-out functions, `\|y\|` = 4–6, `k = 2` |
| **OOD: More Answers** | `k` ∈ [3, 6] |
| **OOD: Longer Outputs** | `\|y\|` = 6–10 |
| **OOD: Longer Outputs & More Answers** | Both shifts combined |
| **OOD: Length Change** | Unseen functions where `\|x\| ≠ \|y\|` |
| **Scoring** | `min(#correct unique, k) / max(k, #submitted)`, verified by executing `f` |

Models are trained from **Qwen2.5-7B-Instruct** with GRPO and DAPO-style filtering of
zero-variance groups. The function code is included in every prompt, so no SFT warm-up
is needed.

- **Mono**: RL on non-bijective inversion only.
- **Mixed**: RL on bijective and non-bijective inversion at a 1:1 ratio.

The training engine is a trimmed copy of [veRL](https://github.com/volcengine/verl)
under `verl/`. All task generation lives under `projects/`.

---

## Environment setup

1. Clone and create a Python environment (Python 3.10+ tested).

   ```bash
   git clone https://github.com/ziqipang/rl_meta_thinking.git
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
├── atomic_functions.py          # bijective function library (func_00..23)
├── generators/                  # task generators
│   ├── create_bijective.py                 # bijective forward / inverse data
│   ├── create_bijective_multi_response.py  # bijective multi-response
│   ├── create_nonbijective.py              # non-bijective function library + base generator
│   ├── create_nonbijective_multi.py        # non-bijective multi-preimage helpers
│   ├── create_multi_response_with_none.py  # non-bijective multi-response
│   └── diverse_evals/                      # OOD evaluation sets
└── pipelines/
    ├── prepare_data.sh                     # generate RL train + eval data
    ├── rl_nonbijective_only.sh             # Mono: GRPO on non-bijective only
    └── rl_both_funcs.sh                    # Mixed: GRPO on bijective + non-bijective

verl/                            # training engine (vendored)
examples/grpo_trainer/template.sh # GRPO entrypoint used by RL scripts
scripts/model_merger.py          # post-RL FSDP shard merger
```

---

## Reproducing Mono and Mixed

### Step 1: generate RL data

```bash
bash projects/pipelines/prepare_data.sh
```

Outputs:
- `data/stage2/{bijective,nonbijective}/{train,test}.parquet`: training data and in-distribution eval
- `data/eval/bijective_inverse/test.parquet`: bijective single-answer inverse eval
- `data/eval/more_candidates/test.parquet`: OOD **More Answers**
- `data/eval/long_outputs/test.parquet`: OOD **Longer Outputs**
- `data/eval/long_outputs_more_candidates/test.parquet`: OOD **Longer Outputs & More Answers**
- `data/eval/lenchange_functions/test.parquet`: OOD **Length Change**

### Step 2: launch GRPO

Each script runs 200 GRPO steps on one node with 8 GPUs (about one day on 8×H100 80GB).
To change locations or scale, edit `NNODES`, `MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`, or `SAVE_DIR`
in the script. The full set of hyperparameters is in `examples/grpo_trainer/template.sh`
and Appendix A.3 of the paper.

```bash
# Mono: RL on non-bijective inversion only
bash projects/pipelines/rl_nonbijective_only.sh

# Mixed: RL co-training on bijective + non-bijective inversion (1:1)
bash projects/pipelines/rl_both_funcs.sh
```

All in-distribution and OOD sets are validated every 20 steps.

### Step 3: merge the FSDP checkpoint (after training)

```bash
python3 scripts/model_merger.py \
    --local_dir disk/checkpoints/reasoning-data-rl/<experiment_name>/global_step_<N>/actor
```

---

## Acknowledgements

This codebase is largely inspired by and builds on
[Yuan et al., *From f(x) and g(x) to f(g(x)): LLMs Learn New Skills in RL by Composing Old Ones*](https://arxiv.org/abs/2509.25123)
(ICLR 2026). Our string-function design, function-level train/held-out split, and
overall controlled-RL setup follow their work. We reverse their task direction, from
computing `f(x)` to inverting it, so that reasoning strategies become observable. We
thank the authors for their open research.

We also thank the [veRL](https://github.com/volcengine/verl) team for the RL training
framework and the [DAPO](https://arxiv.org/abs/2503.14476) authors for their training recipe.

## Citation

If you find this work useful, please cite:

```bibtex
@article{pang2026not,
  title   = {Not All Reasoning Generalizes: A Case Study on Learning Generalizable Reasoning Strategies in RLVR},
  author  = {Pang, Ziqi and Wang, Yu-Xiong},
  journal = {arXiv preprint arXiv:2611.03627},
  year    = {2026}
}
```

Please also consider citing Yuan et al., whose work this codebase builds on:

```bibtex
@inproceedings{yuan2026composing,
  title     = {From $f(x)$ and $g(x)$ to $f(g(x))$: {LLMs} Learn New Skills in {RL} by Composing Old Ones},
  author    = {Yuan, Lifan and Chen, Weize and Zhang, Yuchen and Cui, Ganqu and Wang, Hanbin and You, Ziming and Ding, Ning and Liu, Zhiyuan and Sun, Maosong and Peng, Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
