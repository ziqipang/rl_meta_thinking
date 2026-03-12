#!/usr/bin/env bash
# Full post-training pipeline: merge checkpoints → rollouts → HTMLs
# Run this after BOTH RL experiments have completed.
set -euo pipefail
set -x

cd /home/ubuntu/rl_meta_thinking

BOTH_CKPT=disk/checkpoints/numerical-task-v3-mix-rl/stage2-rl-both-bijective-v3-and-nonbijective-v2
NB_ONLY_CKPT=disk/checkpoints/numerical-task-v3-mix-rl/stage2-rl-nonbijective-v2-only

# ---- Step 1: Merge both_model checkpoint ----
BOTH_ACTOR=${BOTH_CKPT}/global_step_160/actor
if [ ! -f "${BOTH_ACTOR}/huggingface/model-00001-of-00004.safetensors" ]; then
    echo "=== Merging both_model checkpoint ==="
    python3 scripts/model_merger.py --local_dir "${BOTH_ACTOR}"
else
    echo "=== both_model already merged ==="
fi

# Verify nb_only is already merged
NB_ACTOR=${NB_ONLY_CKPT}/global_step_160/actor
if [ ! -f "${NB_ACTOR}/huggingface/model-00001-of-00004.safetensors" ]; then
    echo "=== Merging nb_only_model checkpoint ==="
    python3 scripts/model_merger.py --local_dir "${NB_ACTOR}"
else
    echo "=== nb_only_model already merged ==="
fi

# ---- Step 2: Run rollouts ----
bash analysis/v3_mix/run_all_rollouts.sh

echo "=== Post-training pipeline complete ==="
